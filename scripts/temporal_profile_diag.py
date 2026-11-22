"""TT0 · Diagnóstico del perfil temporal implícito (adaptación de *document focus time*).

Idea (Jatowt, Au Yeung y Tanaka, CIKM 2013; ``papers/jatowt2013_focus_time.pdf``): aprender
del propio corpus qué n-gramas "apuntan" a un periodo, contando con qué décadas co-ocurren
en la misma oración. Con eso, un texto sin ningún año explícito ("Weimar", "Henry VIII",
"SegWit") también tiene un perfil temporal. Aquí se mide, dentro del pool top-100 del
híbrido BM25 + denso, si la similitud entre el perfil de la consulta y el del documento
separa gold de no-gold y si aporta algo al fusionarla con el híbrido.

Es un diagnóstico (usa los qrels de train), no un sistema. Ver RESULTADOS_PERFIL_TEMPORAL.md.

Pasos:
1. **Léxico** (``build_lexicon``): para cada oración del corpus del dominio con alguna fecha
   explícita (``reteco.temporal.extract_spans``), cada unigrama y bigrama suma la oración a
   los bins de tiempo de sus fechas. De ahí salen ``P(bin | n-grama)`` (suavizada hacia el
   fondo ``P0``), la **especificidad** ``KL(P(·|w) || P0)`` (cuánto se concentra en un
   periodo) y la **fechabilidad** (cuánto más que la media aparece en oraciones fechadas).
2. **Perfiles** (``profiles``): explícito (fechas del texto) e implícito (media de
   ``P(·|w)`` de sus 20 n-gramas más específicos, pesada por la especificidad).
3. **Evaluación**: coseno entre perfiles (menos el fondo, suavizados con los bins vecinos);
   AUC por consulta gold / no-gold dentro del pool; y fusión RRF del orden temporal con el
   del híbrido (pesos 0.1, 0.25 y 0.5) evaluada con nDCG@10.

Bins de tiempo: siglos de -1000 a 999 y décadas de 1000 a 2039.

Uso (en el servidor; el conteo usa ``--procs`` procesos):
    .venv/bin/python scripts/temporal_profile_diag.py --domains politics history hsm

Argumentos:
    --domains    dominios a analizar.
    --procs      procesos para el conteo (14).
    --min-dated  oraciones fechadas mínimas para que un n-grama entre en el léxico (10).
    --alpha      fuerza del suavizado de Dirichlet hacia el fondo (5).
    --top-k      n-gramas más específicos que forman el perfil implícito de un texto (20).
    --spec-min   especificidad mínima de esos n-gramas (0.3).
    --depth      profundidad del pool del híbrido (100).
    --doc-words  palabras de cada documento que se usan para su perfil (1500).

Salidas en ``outputs/exp/perfil_temporal/``: ``<dominio>_lex.pkl`` (léxico; se reutiliza si
existe) y ``<dominio>_<subtrack>_stats.pkl`` (tabla de resultados). Imprime el léxico
(top por especificidad y por fechabilidad) y la tabla por subtrack.
"""

import argparse
import json
import math
import pickle
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Permite importar el paquete reteco (en src/) sin instalarlo; por eso los imports que
# siguen llevan "noqa: E402" (import que no está al principio del archivo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco import data, temporal  # noqa: E402
from reteco.evaluate import evaluate_run, load_run  # noqa: E402

# Runs del híbrido BM25 + denso (RRF; BM25 con peso 0.5 y bge-base con peso 1), la mejor
# primera etapa del proyecto (RESULTADOS_EXPERIMENTOS.md §2.8).
RUNS = {
    "1a": "rrf(bm25:rrf(limpia,titulo)+bge-base:rrf(limpia,titulo))w0.5,1",
    "1b": "rrf(bm25:titulo+paso_limpio+bge-base:limpia+paso)w0.5,1",
}
OUT = data.EXP_DIR / "perfil_temporal"

# ----------------------------------------------------------------------------- bins de tiempo
# Siglos antes del año 1000 (de -1000 a 999) y décadas de 1000 a 2039.
CENT0, DEC0, DEC1 = -1000, 1000, 2040
N_CENT = (DEC0 - CENT0) // 100   # 20 bins de siglo
N_DEC = (DEC1 - DEC0) // 10      # 104 bins de década
NB = N_CENT + N_DEC


def year_bin(y: int) -> int | None:
    """Índice del bin de un año, o ``None`` si cae fuera de [-1000, 2040)."""
    if y < CENT0 or y >= DEC1:
        return None
    return (y - CENT0) // 100 if y < DEC0 else N_CENT + (y - DEC0) // 10


def bin_label(k: int) -> str:
    """Etiqueta legible de un bin: ``"1830s"`` para décadas, ``"s.-300"`` para siglos antiguos."""
    return f"s.{CENT0 + 100 * k}" if k < N_CENT else f"{DEC0 + 10 * (k - N_CENT)}s"


def span_bins(span: temporal.TimeSpan) -> dict[int, float]:
    """Reparte un ``TimeSpan`` entre los bins que cubre, con pesos que suman 1.

    Devuelve ``{}`` para la actualidad ("currently") y para intervalos de más de 300
    años, que son demasiado vagos para fechar nada.
    """
    if span.gran == "present" or span.end - span.start > 300:
        return {}
    a = math.floor(span.start)
    b = max(a + 1, math.ceil(span.end))
    c = Counter(k for k in (year_bin(y) for y in range(a, b)) if k is not None)
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()} if tot else {}


# ----------------------------------------------------------------------------- n-gramas
# Palabras funcionales que se descartan. Se conservan a propósito las temporales
# ("before", "after", "during", "until", "since", "when"...) para poder medirlas.
STOP = set("""a about above again against all am an and any are as at be because been being
below between both but by can could did do does doing down each few for from further had has
have having he her here hers herself him himself his how i if in into is it its itself just me
more most my myself no nor not of off on only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves there these they this those
through to too under up very was we were what where which while who whom why will with would you
your yours yourself also may might must shall one us use used using get got like""".split())
TOK = re.compile(r"[a-z][a-z'-]*[a-z]")                       # palabras sin dígitos (los años van aparte)
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|\n+")        # fin de oración o salto de línea
MAYBE_DATE = re.compile(r"\d|centur|millenni", re.I)          # filtro rápido antes del extractor


def ngrams(text_lower: str) -> set[str]:
    """Unigramas y bigramas de un texto en minúsculas, sin funcionales ni números.

    Un bigrama solo se forma con dos palabras de contenido consecutivas en el texto
    original (no se "salta" por encima de las funcionales).
    """
    toks = TOK.findall(text_lower)
    ok = [t not in STOP and len(t) > 2 for t in toks]
    uni = {t for t, o in zip(toks, ok) if o}
    bi = {f"{toks[i]} {toks[i + 1]}" for i in range(len(toks) - 1) if ok[i] and ok[i + 1]}
    return uni | bi


def text_bins(text: str) -> np.ndarray:
    """Masa explícita por bin: cada expresión temporal del texto suma 1, repartido entre sus bins."""
    v = np.zeros(NB)
    if MAYBE_DATE.search(text):
        for _, _, span in temporal.extract_spans(text):
            for k, w in span_bins(span).items():
                v[k] += w
    return v


# ----------------------------------------------------------------------------- léxico (conteo en paralelo)


def count_shard(args: tuple[str, int, int]) -> tuple:
    """Cuenta co-ocurrencias n-grama–bin en las oraciones de un fragmento del corpus.

    Cada worker lee el archivo entero y procesa solo las líneas ``i % n == k``.

    Args:
        args: ``(dominio, k, n)``.

    Returns:
        ``(nw, ndw, cwb, bin_tot, n_sent, n_dated)``: oraciones con cada n-grama (podado a
        ``≥ 2`` para que la fusión sea manejable), oraciones fechadas con cada n-grama,
        masa ``(n-grama, bin)``, masa total por bin, oraciones y oraciones fechadas.
    """
    domain, k, n = args
    nw, ndw, cwb = Counter(), Counter(), Counter()
    bin_tot = np.zeros(NB)
    n_sent = n_dated = 0
    with open(data.T1_DIR / domain / "documents.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % n != k or not line.strip():
                continue
            text = json.loads(line)["content"][:50_000]   # tope para documentos enormes
            for sent in SENT.split(text):
                if len(sent) < 20:
                    continue
                grams = ngrams(sent.lower())
                n_sent += 1
                nw.update(grams)
                v = text_bins(sent)
                tot = v.sum()
                if tot == 0:
                    continue
                v /= tot   # cada oración fechada pesa 1 en total, repartido entre sus bins
                n_dated += 1
                bin_tot += v
                ndw.update(grams)
                nz = np.flatnonzero(v)
                for g in grams:
                    for b in nz:
                        cwb[(g, int(b))] += v[b]
    nw = Counter({g: c for g, c in nw.items() if c >= 2})
    return nw, ndw, cwb, bin_tot, n_sent, n_dated


def build_lexicon(domain: str, procs: int, min_dated: int, alpha: float) -> dict:
    """Construye ``P(bin | n-grama)``, la especificidad y la fechabilidad de un dominio.

    Args:
        domain: dominio.
        procs: procesos para el conteo.
        min_dated: oraciones fechadas mínimas para que un n-grama entre en el léxico.
        alpha: suavizado de Dirichlet: ``P = (C + α·P0) / (n_fechadas + α)``.

    Returns:
        Diccionario con ``vocab`` (lista), ``vid`` (n-grama → fila), ``P`` (matriz
        ``|V| × NB``), ``spec`` (KL frente al fondo, en nats), ``p0`` (fondo), ``nd``
        (oraciones fechadas por n-grama), ``nw`` (oraciones por n-grama), ``dated_lift``
        (fechabilidad) y los totales ``n_sent`` / ``n_dated``.
    """
    t0 = time.time()
    with Pool(procs) as pool:
        parts = pool.map(count_shard, [(domain, k, procs) for k in range(procs)])
    nw, ndw, cwb = Counter(), Counter(), Counter()
    bin_tot = np.zeros(NB)
    n_sent = n_dated = 0
    for a, b, c, bt, ns, nd in parts:
        nw.update(a)
        ndw.update(b)
        cwb.update(c)
        bin_tot += bt
        n_sent += ns
        n_dated += nd
    print(f"[{domain}] conteo {time.time() - t0:.0f}s · oraciones {n_sent:,} · fechadas {n_dated:,} "
          f"({n_dated / max(n_sent, 1):.1%}) · n-gramas fechados {len(ndw):,}", flush=True)

    vocab = sorted(g for g, c in ndw.items() if c >= min_dated)
    vid = {g: i for i, g in enumerate(vocab)}
    C = np.zeros((len(vocab), NB), dtype=np.float32)
    for (g, b), c in cwb.items():
        i = vid.get(g)
        if i is not None:
            C[i, b] += c
    p0 = bin_tot / bin_tot.sum()   # fondo: distribución de las oraciones fechadas del dominio
    nd = np.array([ndw[g] for g in vocab], dtype=np.float32)
    P = (C + alpha * p0) / (nd[:, None] + alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.nansum(P * np.log(P / p0), axis=1)
    # Fechabilidad: P(oración fechada | w) / P(oración fechada). Un n-grama que solo se ve en
    # oraciones fechadas (nw < 2 tras la poda) usa ndw como denominador.
    base = n_dated / n_sent
    dated_lift = np.array([(ndw[g] / max(nw.get(g, ndw[g]), 1)) / base for g in vocab])
    return dict(vocab=vocab, vid=vid, P=P, spec=spec, p0=p0, nd=nd, dated_lift=dated_lift,
                nw=np.array([nw.get(g, 0) for g in vocab]), n_sent=n_sent, n_dated=n_dated)


# ----------------------------------------------------------------------------- perfiles

KERNEL = np.array([0.25, 0.5, 0.25])   # suavizado con los bins vecinos


def smooth(v: np.ndarray) -> np.ndarray:
    """Suaviza un perfil con los bins vecinos, para que 1834 y "the 1840s" no sean ortogonales."""
    return np.convolve(v, KERNEL, mode="same")


def profiles(text: str, lex: dict, top_k: int, spec_min: float) -> dict:
    """Perfil explícito e implícito de un texto.

    Args:
        text: consulta o documento.
        lex: léxico de ``build_lexicon``.
        top_k: n-gramas más específicos que forman el perfil implícito.
        spec_min: especificidad mínima para que un n-grama cuente.

    Returns:
        ``{"exp": distribución por bin de las fechas explícitas o None, "imp": media de
        P(·|w) de los n-gramas elegidos (pesada por especificidad) o None, "words": los 8
        n-gramas que más pesan}``.
    """
    e = text_bins(text)
    grams = [g for g in ngrams(text.lower()) if g in lex["vid"]]
    idx = np.array([lex["vid"][g] for g in grams], dtype=int)
    imp = None
    words = []
    if len(idx):
        s = lex["spec"][idx]
        keep = s >= spec_min
        idx, s = idx[keep], s[keep]
        if len(idx):
            order = np.argsort(-s)[:top_k]
            idx, s = idx[order], s[order]
            imp = (s[:, None] * lex["P"][idx]).sum(0) / s.sum()
            words = [lex["vocab"][i] for i in idx[:8]]
    exp = e / e.sum() if e.sum() > 0 else None
    return dict(exp=exp, imp=imp, words=words)


def deviation(dist: np.ndarray | None, p0: np.ndarray) -> np.ndarray | None:
    """Perfil suavizado menos el fondo, normalizado a norma 1 (para el coseno).

    Restar el fondo evita que todos los perfiles se parezcan por estar concentrados en
    los siglos XIX–XXI, que es donde cae la mayoría de las fechas de cualquier dominio.
    """
    if dist is None:
        return None
    d = smooth(dist) - smooth(p0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def mix(p: dict, which: str) -> np.ndarray | None:
    """Combina perfiles: ``E`` (solo explícito), ``I`` (solo implícito) o ``EI`` (mitad y mitad).

    En ``EI``, si solo existe uno de los dos, se usa ese.
    """
    if which == "E":
        return p["exp"]
    if which == "I":
        return p["imp"]
    parts = [x for x in (p["exp"], p["imp"]) if x is not None]
    return sum(parts) / len(parts) if parts else None


# ----------------------------------------------------------------------------- evaluación


def auc(pos: list[float], neg: list[float]) -> float:
    """AUC de Mann-Whitney: probabilidad de que un gold puntúe más que un no-gold (empates a 0.5)."""
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def query_texts(domain: str, subtrack: str) -> dict[str, str]:
    """Texto de cada consulta (1a: ``limpia``) o paso (1b: instrucción del paso + ``limpia``)."""
    if subtrack == "1a":
        return {r["id"]: data.clean_query(r["query"]) for r in data.load_examples(domain, "train")}
    out = {}
    for r in data.load_steps(domain, "train"):
        q = data.clean_query(r["query"])
        for s in r["steps"]:
            out[s["step_id"]] = f"{s['step_instruction']} {q}"
    return out


def print_lexicon(dom: str, lex: dict, min_dated: int) -> None:
    """Imprime los n-gramas más específicos (de un periodo) y los más fechables (marcadores)."""
    V, spec, P = lex["vocab"], lex["spec"], lex["P"]
    print(f"\n[{dom}] vocabulario fechado (≥{min_dated} oraciones fechadas): {len(V):,}")
    print("spec: percentiles 50/90/99 =", np.round(np.percentile(spec, [50, 90, 99]), 2))
    print("\nTop 40 por especificidad (KL frente al fondo) — palabras 'de un periodo':")
    for i in np.argsort(-spec)[:40]:
        print(f"  {V[i]:<28} spec {spec[i]:.2f}  pico {bin_label(int(P[i].argmax())):>8}  n_fech {int(lex['nd'][i])}")
    ok = lex["nw"] >= 50   # la fechabilidad solo es fiable con frecuencia suficiente
    dl = np.where(ok, lex["dated_lift"], 0)
    print("\nTop 30 por 'fechabilidad' (P(oración fechada | w) / base, Nw≥50) — marcadores temporales:")
    for i in np.argsort(-dl)[:30]:
        print(f"  {V[i]:<28} lift {dl[i]:.2f}  spec {spec[i]:.2f}  Nw {lex['nw'][i]}")


def evaluate_subtrack(dom: str, sub: str, lex: dict, args: argparse.Namespace) -> dict:
    """Separación gold / no-gold y fusión con el híbrido para un dominio y subtrack.

    Returns:
        ``{perfil: métricas}`` para ``E``, ``I`` y ``EI``: cobertura (consultas con perfil),
        AUC temporal media, AUC en las consultas sin fecha explícita, AUC del propio
        híbrido en el pool, similitud media gold / no-gold, nDCG@10 de la base y de la
        fusión con cada peso.
    """
    run = load_run("fusion", RUNS[sub], dom, "train")
    qrels = data.load_qrels(dom, "train", sub)
    texts = query_texts(dom, sub)
    pools = {q: sorted(r, key=r.get, reverse=True)[: args.depth] for q, r in run.items() if q in qrels}
    need = {d for p in pools.values() for d in p}
    doc_txt = {d["id"]: " ".join(d["content"].split()[: args.doc_words])
               for d in data.iter_documents(dom) if d["id"] in need}
    t0 = time.time()
    dprof = {d: profiles(t, lex, args.top_k, args.spec_min) for d, t in doc_txt.items()}
    print(f"\n[{dom} {sub}] perfiles de {len(dprof):,} documentos del pool en {time.time() - t0:.0f}s")
    qprof = {q: profiles(texts[q], lex, args.top_k, args.spec_min) for q in pools}

    p0 = lex["p0"]
    # El run base es el pool con scores RRF de su posición: el nDCG@10 coincide con el del híbrido.
    base_run = {q: {d: 1.0 / (60 + r) for r, d in enumerate(p, 1)} for q, p in pools.items()}
    nd_base = evaluate_run(base_run, qrels)["ndcg_cut_10"].mean()
    stats, examples = {}, []
    for which in ("E", "I", "EI"):
        aucs, aucs_h, aucs_noyear, gold_sims, ng_sims, cov = [], [], [], [], [], 0
        new_runs = {w: {} for w in (0.1, 0.25, 0.5)}
        for q, pool in pools.items():
            gold = {d for d, r in qrels[q].items() if r > 0}
            qv = deviation(mix(qprof[q], which), p0)
            base = base_run[q]
            if qv is None:   # consulta sin perfil: se deja el orden del híbrido
                for w in new_runs:
                    new_runs[w][q] = base
                continue
            cov += 1
            sims = {}
            for d in pool:
                dv = deviation(mix(dprof[d], which), p0) if d in dprof else None
                sims[d] = float(qv @ dv) if dv is not None else 0.0
            # Orden temporal: por similitud; los empates conservan el orden del híbrido.
            srank = {d: r for r, d in enumerate(sorted(pool, key=lambda d: -sims[d]), 1)}
            for w in new_runs:
                new_runs[w][q] = {d: base[d] + w / (60 + srank[d]) for d in pool}
            pos = [sims[d] for d in pool if d in gold]
            neg = [sims[d] for d in pool if d not in gold]
            if pos and neg:
                aucs.append(auc(pos, neg))
                aucs_h.append(auc([base[d] for d in pool if d in gold], [base[d] for d in pool if d not in gold]))
                if qprof[q]["exp"] is None:
                    aucs_noyear.append(aucs[-1])
                gold_sims += pos
                ng_sims += neg
                if which == "EI" and len(examples) < 6 and aucs[-1] > 0.8 and qprof[q]["exp"] is None:
                    examples.append((q, qprof[q]["words"], texts[q][:120]))
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
        stats[which] = dict(cov=cov / len(pools), n_auc=len(aucs), auc=mean(aucs), auc_h=mean(aucs_h),
                            auc_noyear=mean(aucs_noyear), n_noyear=len(aucs_noyear),
                            sim_gold=mean(gold_sims), sim_ng=mean(ng_sims), nd_base=nd_base,
                            **{f"nd_w{w}": evaluate_run(r, qrels)["ndcg_cut_10"].mean() for w, r in new_runs.items()})
    print(f"[{dom} {sub}] consultas con pool: {len(pools)}")
    print(f"{'perfil':<5} {'cobert':>7} {'n':>4} {'AUC tmp':>8} {'AUC sin año':>12} {'(n)':>4} "
          f"{'AUC híbrido':>11} {'sim g/ng':>12} {'nDCG base':>9} {'w0.1':>7} {'w0.25':>7} {'w0.5':>7}")
    for which, s in stats.items():
        print(f"{which:<5} {s['cov']:>7.2f} {s['n_auc']:>4} {s['auc']:>8.3f} {s['auc_noyear']:>12.3f} "
              f"{s['n_noyear']:>4} {s['auc_h']:>11.3f} {s['sim_gold']:>5.3f}/{s['sim_ng']:<6.3f} "
              f"{s['nd_base']:>9.4f} {s['nd_w0.1']:>7.4f} {s['nd_w0.25']:>7.4f} {s['nd_w0.5']:>7.4f}")
    if examples:
        print("Ejemplos (consulta sin año, AUC temporal > 0.8): n-gramas que dan el perfil")
        for q, words, t in examples:
            print(f"  {q}: {words} | {t!r}")
    return stats


def main() -> None:
    """Construye (o carga) el léxico de cada dominio, lo imprime y evalúa 1a y 1b."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--procs", type=int, default=14)
    parser.add_argument("--min-dated", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--spec-min", type=float, default=0.3)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--doc-words", type=int, default=1500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for dom in args.domains:
        lex_path = OUT / f"{dom}_lex.pkl"
        if lex_path.exists():
            lex = pickle.loads(lex_path.read_bytes())
        else:
            lex = build_lexicon(dom, args.procs, args.min_dated, args.alpha)
            lex_path.write_bytes(pickle.dumps(lex))
        print_lexicon(dom, lex, args.min_dated)
        for sub in ("1a", "1b"):
            stats = evaluate_subtrack(dom, sub, lex, args)
            (OUT / f"{dom}_{sub}_stats.pkl").write_bytes(pickle.dumps(stats))


if __name__ == "__main__":
    main()
