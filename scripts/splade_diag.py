"""Diagnósticos de SPLADE (PLAN_SPLADE.md, SP5). Solo describen: no deciden nada ni escriben en resultados.csv.

Subcomandos:
    expansiones  Qué términos activa SPLADE en las consultas: cuánto peso va a términos que
                 no están en el texto (expansión), qué términos temporales añade y ejemplos
                 estratificados por clase de razonamiento.
    anios        Los años en el vocabulario de BERT, el peso que reciben en la consulta, los
                 años que añade la expansión y si los gold activan el año de la consulta más
                 que los no-gold del top-100 de SPLADE (como el lift de TG0).
    grupos       nDCG@10 por clase de razonamiento y por intención para varios runs.
    ejemplos     Gold que el run A pone en su top-10 y el run B ni siquiera en su top-100.

Uso (en el servidor; los dos primeros usan la GPU para codificar consultas):
    .venv/bin/python scripts/splade_diag.py expansiones --subtrack 1a --variant limpia
    .venv/bin/python scripts/splade_diag.py anios --subtrack 1a --variant limpia
    .venv/bin/python scripts/splade_diag.py grupos --subtrack 1a --runs SPLADE=splade-ed/limpia "Denso=bge-base/rrf(limpia,titulo)"
    .venv/bin/python scripts/splade_diag.py ejemplos --subtrack 1a --runs splade-ed/limpia "bge-base/rrf(limpia,titulo)" --variant titulo

Las etiquetas de clase e intención salen de ``guidance``: aquí solo se usan para analizar,
como en el resto del proyecto. Todo es sobre train.

Salida: tablas en pantalla y un ``.md`` por diagnóstico en ``outputs/exp/splade/diag/``.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Permite importar el paquete reteco (en src/) sin instalarlo; por eso los imports que
# siguen llevan "noqa: E402" (import que no está al principio del archivo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.data import (  # noqa: E402
    DOMAINS,
    EXP_DIR,
    SEED,
    iter_documents,
    load_qrels,
    parent_query_id,
    query_labels,
)
from reteco.evaluate import by_group, evaluate_variant, has_run, load_run, macro  # noqa: E402
from reteco.queries import VARIANTS  # noqa: E402

DIAG_DIR = EXP_DIR / "splade" / "diag"

# Un año "histórico o moderno" escrito con cuatro cifras (1000–2029): son los que pueden ser
# un único token en el vocabulario de BERT.
YEAR = re.compile(r"^(1[0-9]{3}|20[0-2][0-9])$")
# Ordinales como "19th" (de "19th century").
ORDINAL = re.compile(r"^\d{1,2}(st|nd|rd|th)$")
# Décadas escritas como un solo token del vocabulario ("1930s").
DECADE = re.compile(r"^(1[0-9]|20)[0-9]0s$")
# Palabras temporales (en minúsculas, como el vocabulario uncased). Lista corta y explícita:
# periodos, fechas, orden y vigencia; sirve para contar, no para clasificar.
TIME_WORDS = {
    "century", "centuries", "decade", "decades", "era", "eras", "epoch", "period", "periods",
    "year", "years", "annual", "date", "dates", "dated", "dating", "ancient", "medieval", "modern",
    "contemporary", "historical", "historic", "history", "antiquity", "prehistoric", "renaissance",
    "victorian", "colonial", "bc", "bce", "ad", "age", "ages", "reign", "dynasty", "before", "after",
    "during", "since", "until", "when", "earliest", "earlier", "first", "latest", "later", "oldest",
    "older", "recent", "recently", "current", "currently", "today", "now", "early", "late",
    "former", "formerly", "originally", "origin", "origins", "timeline", "chronology",
}


def is_year(tok: str) -> bool:
    """¿El token es un año de cuatro cifras entre 1000 y 2029?"""
    return bool(YEAR.match(tok))


def is_temporal(tok: str) -> bool:
    """¿El token es un año, una década (``1930s``), un ordinal (``19th``) o una palabra de ``TIME_WORDS``?"""
    return is_year(tok) or bool(DECADE.match(tok)) or bool(ORDINAL.match(tok)) or tok in TIME_WORDS


def ordinal(n: int) -> str:
    """Ordinal en inglés: 1 → ``1st``, 12 → ``12th``, 19 → ``19th``, 21 → ``21st``."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def write_md(name: str, lines: list[str]) -> Path:
    """Escribe un informe Markdown en ``outputs/exp/splade/diag/<name>.md`` y devuelve la ruta."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAG_DIR / f"{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def md_table(df: pd.DataFrame, fmt: str = "{:.4f}") -> list[str]:
    """Tabla Markdown sencilla (sin depender de ``tabulate``).

    Args:
        df: tabla; el índice es la primera columna.
        fmt: formato de los números con decimales.

    Returns:
        Líneas de la tabla.
    """
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else fmt.format(v)
        return str(v)
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    # itertuples conserva el tipo de cada columna (iterrows pasaría los recuentos a float).
    for idx, row in zip(df.index, df.itertuples(index=False)):
        out.append("| " + " | ".join([str(idx)] + [cell(v) for v in row]) + " |")
    return out


def load_queries(subtrack: str, variant: str, domains: list[str]) -> list[tuple[str, str, str]]:
    """Consultas de train de una variante.

    Args:
        subtrack: ``"1a"`` o ``"1b"``.
        variant: variante de ``reteco.queries.VARIANTS``; si da varios textos, se usa el primero.
        domains: dominios.

    Returns:
        ``[(dominio, qid, texto)]``.
    """
    out = []
    for dom in domains:
        for qid, q in VARIANTS[subtrack][variant].build(dom, "train").items():
            out.append((dom, qid, q[0] if isinstance(q, list) else q))
    return out


def encode_queries(key: str, texts: list[str]):
    """Codifica consultas con el modelo SPLADE y devuelve también sus tokens.

    Usa torch: se importa aquí para que ``grupos`` y ``ejemplos`` no lo carguen.

    Args:
        key: clave de ``reteco.splade.MODELS``.
        texts: textos.

    Returns:
        ``(CSR consultas × V, [conjunto de ids de token de cada consulta], vocabulario id → token, tokenizador)``.
    """
    from reteco import splade
    from reteco.dense import QUERY_MAX_TOKENS
    model = splade.get_model(key)
    model.max_seq_length = QUERY_MAX_TOKENS
    mat = splade.encode_texts(model, texts, batch_size=32, kind="query")
    # Tokens propios de cada consulta, con la misma truncación que al codificar y sin [CLS]/[SEP].
    own = model.tokenizer(texts, add_special_tokens=False, truncation=True, max_length=QUERY_MAX_TOKENS)["input_ids"]
    vocab = model.tokenizer.convert_ids_to_tokens(list(range(mat.shape[1])))
    return mat, [set(o) for o in own], vocab, model.tokenizer


def label_of(dom: str, qid: str) -> dict[str, str]:
    """Clase e intención (de ``guidance``) de la consulta madre de ``qid``; ``other`` si falta."""
    return query_labels(dom, "train").get(parent_query_id(qid), {"clase": "other", "intent": "other"})


# ----------------------------------------------------------------------------- expansiones

def cmd_expansiones(args) -> None:
    """Cuánto pesa la expansión, qué términos temporales añade y ejemplos por clase."""
    queries = load_queries(args.subtrack, args.variant, args.domains)
    mat, own, vocab, _ = encode_queries(args.model, [t for _, _, t in queries])
    rows, exp_top, exp_time, exp_years = [], Counter(), Counter(), Counter()
    tops = []   # (términos de más peso de cada consulta) para los ejemplos
    for i, (dom, qid, text) in enumerate(queries):
        lo, hi = mat.indptr[i], mat.indptr[i + 1]
        cols, w = mat.indices[lo:hi], mat.data[lo:hi]
        toks = [vocab[c] for c in cols]
        is_exp = np.array([c not in own[i] for c in cols], dtype=bool)   # término que no está en el texto
        order = np.argsort(-w)[:args.top]
        exp_toks = [t for t, e in zip(toks, is_exp) if e]
        exp_toks_top = [toks[j] for j in order if is_exp[j]]
        exp_top.update(exp_toks_top)
        exp_time.update(t for t in exp_toks_top if is_temporal(t) and not is_year(t))
        exp_years.update(t for t in exp_toks if is_year(t))
        lab = label_of(dom, qid)
        rows.append({
            "domain": dom, "clase": lab["clase"], "intent": lab["intent"],
            "terminos": len(cols),
            "frac_exp_terminos": float(is_exp.mean()) if len(cols) else 0.0,
            "frac_exp_peso": float(w[is_exp].sum() / w.sum()) if w.sum() > 0 else 0.0,
            f"frac_exp_top{args.top}": float(is_exp[order].mean()) if len(order) else 0.0,
            "anade_anio": any(is_year(t) for t in exp_toks),
            f"anade_temporal_top{args.top}": any(is_temporal(t) for t in exp_toks_top),
        })
        tops.append([(toks[j], float(w[j]), bool(is_exp[j])) for j in order])
    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c not in ("domain", "clase", "intent")]
    overall = df[metrics].mean().to_frame("todas").T
    by_class = df.groupby("clase")[metrics].mean()
    by_class.insert(0, "n", df.groupby("clase").size())
    overall.insert(0, "n", len(df))
    table = pd.concat([overall, by_class])
    table["n"] = table["n"].astype(int)   # concat lo pasa a float; es un recuento
    table.index.name = "clase"

    lines = [f"# Expansiones de SPLADE · {args.model} · {args.subtrack} `{args.variant}` (train)", "",
             ("Medias por consulta. `frac_exp_*`: fracción de términos, de peso o del top-k que **no** están en "
             "el texto de la consulta (expansión). `anade_anio`: la consulta activa algún año que no escribió. "
             f"`anade_temporal_top{args.top}`: entre sus {args.top} términos de más peso hay un término temporal añadido."), ""]
    lines += md_table(table, "{:.3f}")
    lines += ["", f"## Expansiones más frecuentes en el top-{args.top}", "",
              ", ".join(f"`{t}` ({n})" for t, n in exp_top.most_common(40)), "",
              f"## Términos temporales (no años) añadidos en el top-{args.top}", "",
              ", ".join(f"`{t}` ({n})" for t, n in exp_time.most_common(30)) or "ninguno", "",
              "## Años añadidos (cualquier peso)", "",
              ", ".join(f"`{t}` ({n})" for t, n in exp_years.most_common(30)) or "ninguno", "",
              f"## Ejemplos ({args.n}, estratificados por clase)", "",
              "`+término`: expansión (no está en el texto); ⏱: término temporal.", ""]
    # Muestra estratificada: se recorren las clases por turnos y se toma una consulta al azar de cada una.
    rng = np.random.default_rng(SEED)
    pools = {c: list(rng.permutation(np.flatnonzero(df["clase"].to_numpy() == c))) for c in sorted(df["clase"].unique())}
    picked = []
    while len(picked) < min(args.n, len(df)):
        for c in list(pools):
            if pools[c] and len(picked) < args.n:
                picked.append(pools[c].pop())
    for i in picked:
        dom, qid, text = queries[i]
        terms = " · ".join(f"{'+' if e else ''}{t}{'⏱' if is_temporal(t) else ''} {w:.2f}" for t, w, e in tops[i])
        lines += [(f"- **{dom}** `{qid}` · {df.at[i, 'clase']} / {df.at[i, 'intent']} — "
                  f"“{' '.join(text.split())[:160]}…”"), f"  - {terms}"]
    path = write_md(f"expansiones_{args.model}_{args.subtrack}_{args.variant}", lines)
    print("\n".join(md_table(table, "{:.3f}")))
    print(f"\nExpansiones más frecuentes: {', '.join(t for t, _ in exp_top.most_common(25))}")
    print(f"Temporales añadidos: {', '.join(f'{t} ({n})' for t, n in exp_time.most_common(15))}")
    print(f"→ {path}")


# ----------------------------------------------------------------------------- años

def doc_activation(mat, doc_starts: np.ndarray, n_chunks: int, doc_rows: list[int], cols: list[int]) -> np.ndarray:
    """¿Activa cada documento alguno de los términos ``cols`` en alguno de sus trozos?

    Args:
        mat: CSR trozos × V del dominio.
        doc_starts: primer trozo de cada documento.
        n_chunks: número total de trozos.
        doc_rows: índices (en el corpus) de los documentos a mirar.
        cols: ids de término.

    Returns:
        Booleano por documento de ``doc_rows``.
    """
    ends = np.r_[doc_starts[1:], n_chunks]
    rows = np.concatenate([np.arange(doc_starts[d], ends[d]) for d in doc_rows])
    hit = np.asarray((mat[rows][:, cols] > 0).sum(axis=1)).ravel() > 0
    # Los trozos de cada documento son contiguos en ``rows``: se agrupan con reduceat.
    starts = np.r_[0, np.cumsum([ends[d] - doc_starts[d] for d in doc_rows])[:-1]]
    return np.logical_or.reduceat(hit, starts)


def cmd_anios(args) -> None:
    """Años en el vocabulario, su peso en las consultas y su activación en gold frente a no-gold."""
    from reteco import splade
    queries = load_queries(args.subtrack, args.variant, args.domains)
    mat, own, vocab, tok = encode_queries(args.model, [t for _, _, t in queries])
    tok_vocab = tok.get_vocab()
    single = [y for y in range(1000, 2030) if str(y) in tok_vocab]
    # Tramos de años que NO son un único token (para describir los huecos).
    missing = sorted(set(range(1000, 2030)) - set(single))
    gaps, start = [], None
    for y in missing:
        if start is None:
            start = prev = y
        elif y != prev + 1:
            gaps.append((start, prev))
            start = y
        prev = y
    if start is not None:
        gaps.append((start, prev))
    year_id = {c for c in range(len(vocab)) if is_year(vocab[c])}

    # (b) y (c): peso y rango de los años propios; años añadidos y su distancia al año propio más cercano.
    per_q = []
    for i, (dom, qid, _) in enumerate(queries):
        own_years = sorted(c for c in own[i] if c in year_id)
        if not own_years:
            continue
        lo, hi = mat.indptr[i], mat.indptr[i + 1]
        cols, w = mat.indices[lo:hi], mat.data[lo:hi]
        weight = dict(zip(cols.tolist(), w.tolist()))
        ranks = [1 + int((w > weight.get(c, 0.0)).sum()) if c in weight else None for c in own_years]
        added = [int(vocab[c]) for c in cols if c in year_id and c not in own[i]]
        own_vals = [int(vocab[c]) for c in own_years]
        dist = [min(abs(a - o) for o in own_vals) for a in added]
        # ¿Añade SPLADE la década ("1834" → "1830s") o el siglo ("19th") del año, sin que estén
        # en el texto? Es la normalización de anclas que proponía K4 (PLAN_EXPERIMENTOS §6.2).
        # Solo cuentan las décadas y siglos que existen como token.
        dec_ids = {tok_vocab[f"{y // 10 * 10}s"] for y in own_vals if f"{y // 10 * 10}s" in tok_vocab}
        cent_ids = {tok_vocab[ordinal(y // 100 + 1)] for y in own_vals if ordinal(y // 100 + 1) in tok_vocab}
        dec_new, cent_new = dec_ids - own[i], cent_ids - own[i]
        per_q.append({"domain": dom, "qid": qid, "own_cols": own_years,
                      "activo": np.mean([r is not None for r in ranks]),
                      "rango_mediano": float(np.median([r for r in ranks if r is not None])) if any(ranks) else np.nan,
                      "en_top20": np.mean([r is not None and r <= 20 for r in ranks]),
                      "peso_medio": float(np.mean([weight.get(c, 0.0) for c in own_years])),
                      "anade_anios": len(added) > 0, "n_anadidos": len(added),
                      "dist_mediana": float(np.median(dist)) if dist else np.nan,
                      # NaN si la década o el siglo ya estaban en el texto (o no son un token).
                      "anade_decada": any(c in weight for c in dec_new) if dec_new else np.nan,
                      "anade_siglo": any(c in weight for c in cent_new) if cent_new else np.nan})
    pq = pd.DataFrame(per_q)

    # (d) Activación del año de la consulta en gold y en no-gold del top-100 de SPLADE.
    lift_rows = []
    by_dom = defaultdict(list)
    for r in per_q:
        by_dom[r["domain"]].append(r)
    for dom, items in by_dom.items():
        path = splade.SPLADE_DIR / args.model / f"{dom}.npz"
        if not (path.exists() and has_run(args.model, args.variant, dom, "train")):
            print(f"{dom}: sin índice o sin run {args.model}/{args.variant}; se omite la parte de documentos")
            continue
        dmat, chunk_doc, doc_ids = splade.load_matrix(path)
        doc_starts = np.flatnonzero(np.r_[True, chunk_doc[1:] != chunk_doc[:-1]])
        pos = {d: i for i, d in enumerate(doc_ids)}
        run, qrels = load_run(args.model, args.variant, dom, "train"), load_qrels(dom, "train", args.subtrack)
        for r in items:
            gold = [d for d, rel in qrels.get(r["qid"], {}).items() if rel > 0 and d in pos]
            top = [d for d, _ in sorted(run.get(r["qid"], {}).items(), key=lambda x: -x[1])[:100]]
            nongold = [d for d in top if d not in set(gold)]
            if not gold or not nongold:
                continue
            g = doc_activation(dmat, doc_starts, dmat.shape[0], [pos[d] for d in gold], r["own_cols"])
            n = doc_activation(dmat, doc_starts, dmat.shape[0], [pos[d] for d in nongold], r["own_cols"])
            lift_rows.append({"domain": dom, "gold": g.mean(), "nogold_top100": n.mean()})
        del dmat
    lift = pd.DataFrame(lift_rows)

    lines = [f"# Años en SPLADE · {args.model} · {args.subtrack} `{args.variant}` (train)", "",
             f"**Vocabulario:** {len(single)} de 1,030 años (1000–2029) son un único token. Huecos: "
             + (", ".join(f"{a}–{b}" if a != b else f"{a}" for a, b in gaps) or "ninguno") + ".", "",
             f"**Consultas con algún año de un solo token:** {len(pq)} de {len(queries)} ({len(pq) / max(len(queries), 1):.0%}).", ""]
    if len(pq):
        lines += [(f"- El año de la consulta está activo en su vector: {pq['activo'].mean():.0%}; entre sus 20 términos "
                  f"de más peso: {pq['en_top20'].mean():.0%}; rango mediano {pq['rango_mediano'].median():.0f}; "
                  f"peso medio {pq['peso_medio'].mean():.2f}."),
                  (f"- Consultas en las que SPLADE añade otros años: {pq['anade_anios'].mean():.0%} "
                  f"(mediana de años añadidos {pq.loc[pq['anade_anios'], 'n_anadidos'].median():.0f}; distancia mediana "
                  f"al año propio más cercano {pq['dist_mediana'].median():.0f} años)."),
                  (f"- Activa la década del año sin que esté en el texto (p. ej. 1834 → `1830s`): "
                   f"{pq['anade_decada'].astype(float).mean():.0%} de {int(pq['anade_decada'].notna().sum())} consultas; "
                   f"el siglo (`19th`): {pq['anade_siglo'].astype(float).mean():.0%} de "
                   f"{int(pq['anade_siglo'].notna().sum())}.")]
    if len(lift):
        macro_dom = lift.groupby("domain")[["gold", "nogold_top100"]].mean()
        lines += ["", ("**¿Activan los gold el año de la consulta más que los no-gold del top-100?** "
                  "(media por consulta; macro = media de dominios)"), ""]
        tab = macro_dom.copy()
        tab["lift"] = tab["gold"] / tab["nogold_top100"]
        tab.insert(0, "consultas", lift.groupby("domain").size())
        tab.loc["macro"] = [len(lift), macro_dom["gold"].mean(), macro_dom["nogold_top100"].mean(),
                            macro_dom["gold"].mean() / macro_dom["nogold_top100"].mean()]
        tab["consultas"] = tab["consultas"].astype(int)   # .loc los pasa a float; es un recuento
        tab.index.name = "dominio"
        lines += md_table(tab, "{:.3f}")
    path = write_md(f"anios_{args.model}_{args.subtrack}_{args.variant}", lines)
    print("\n".join(lines))
    print(f"→ {path}")


# ----------------------------------------------------------------------------- grupos

def parse_runs(specs: list[str]) -> list[tuple[str, str, str]]:
    """``["alias=retriever/variante", ...]`` → ``[(alias, retriever, variante)]``; sin alias, el alias es el run."""
    out = []
    for s in specs:
        alias, _, run = s.rpartition("=") if "=" in s.split("/", 1)[0] else ("", "", s)
        ret, var = run.split("/", 1)
        out.append((alias or run, ret, var))
    return out


def cmd_grupos(args) -> None:
    """nDCG@10 medio por clase y por intención de cada run (más la macro por dominio)."""
    runs = parse_runs(args.runs)
    frames = {alias: evaluate_variant(ret, var, args.subtrack) for alias, ret, var in runs}
    first = next(iter(frames.values()))
    lines = [f"# nDCG@10 por grupo · {args.subtrack} (train)", "",
             ("Media por consulta dentro de cada grupo (no macro por dominio); la fila `macro` sí es la macro "
             "del proyecto. Etiquetas de `guidance`, solo para analizar."), ""]
    for col in ("clase", "intent"):
        tab = pd.DataFrame({alias: by_group(pq, col) for alias, pq in frames.items()})
        tab.insert(0, "n", first.groupby(col).size())
        tab.loc["macro"] = [len(first)] + [macro(pq) for pq in frames.values()]
        tab["n"] = tab["n"].astype(int)
        tab.index.name = col
        lines += [f"## Por {col}", ""] + md_table(tab) + [""]
    path = write_md(f"grupos_{args.subtrack}_{args.name}", lines)
    print("\n".join(lines))
    print(f"→ {path}")


# ----------------------------------------------------------------------------- ejemplos

def ranked(ranking: dict[str, float]) -> list[str]:
    """Ids de un ranking ``{doc: score}`` ordenados por score (estable en empates)."""
    return [d for d, _ in sorted(ranking.items(), key=lambda x: -x[1])]


def cmd_ejemplos(args) -> None:
    """Gold en el top-10 de A y fuera del top-100 de B: cuántos hay y una muestra con texto."""
    (alias_a, ret_a, var_a), (alias_b, ret_b, var_b) = parse_runs(args.runs)
    cases = []
    for dom in args.domains:
        if not (has_run(ret_a, var_a, dom, "train") and has_run(ret_b, var_b, dom, "train")):
            continue
        run_a, run_b = load_run(ret_a, var_a, dom, "train"), load_run(ret_b, var_b, dom, "train")
        for qid, rels in load_qrels(dom, "train", args.subtrack).items():
            ra, rb = ranked(run_a.get(qid, {})), ranked(run_b.get(qid, {}))
            pos_b = {d: i + 1 for i, d in enumerate(rb)}
            for d, rel in rels.items():
                if rel > 0 and d in ra[:10] and pos_b.get(d, 10**6) > 100:
                    cases.append((dom, qid, d, ra.index(d) + 1, pos_b.get(d)))
    n_gold_a10 = len(cases)
    rng = np.random.default_rng(SEED)
    sample = [cases[i] for i in sorted(rng.choice(len(cases), size=min(args.n, len(cases)), replace=False))] if cases else []
    # Textos: la consulta (en la variante pedida) y el principio del documento gold.
    want = defaultdict(set)
    for dom, _, d, _, _ in sample:
        want[dom].add(d)
    snippets = {}
    for dom, ids in want.items():
        for doc in iter_documents(dom):
            if doc["id"] in ids:
                text = re.sub(r"<[^>]+>", " ", doc["content"])   # fuera etiquetas HTML
                snippets[doc["id"]] = " ".join(text.split())[:args.chars]
    texts = {}
    for dom in want:
        for qid, q in VARIANTS[args.subtrack][args.variant].build(dom, "train").items():
            texts[(dom, qid)] = q[0] if isinstance(q, list) else q
    lines = [f"# Gold en el top-10 de `{alias_a}` y fuera del top-100 de `{alias_b}` · {args.subtrack} (train)", "",
             f"Casos (consulta, gold): **{n_gold_a10}**. Muestra de {len(sample)}:", ""]
    for dom, qid, d, ra, rb in sample:
        lines += [f"- **{dom}** `{qid}` — consulta: “{' '.join(texts.get((dom, qid), '').split())[:200]}…”",
                  f"  - gold `{d}`: puesto {ra} en A, {'>1000' if rb is None else rb} en B",
                  f"  - “{snippets.get(d, '')}…”"]
    path = write_md(f"ejemplos_{args.subtrack}_{args.name}", lines)
    print("\n".join(lines))
    print(f"→ {path}")


def main() -> None:
    """Interpreta el subcomando y lo ejecuta."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("expansiones", "anios"):
        p = sub.add_parser(name)
        p.add_argument("--subtrack", choices=["1a", "1b"], required=True)
        p.add_argument("--variant", required=True)
        p.add_argument("--model", default="splade-ed")
        p.add_argument("--domains", nargs="*", default=DOMAINS)
        p.add_argument("--top", type=int, default=20, help="términos de más peso que se miran")
        p.add_argument("--n", type=int, default=30, help="ejemplos (solo expansiones)")
    p = sub.add_parser("grupos")
    p.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    p.add_argument("--runs", nargs="+", required=True, help="[alias=]retriever/variante")
    p.add_argument("--name", default="runs", help="sufijo del archivo de salida")
    p = sub.add_parser("ejemplos")
    p.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    p.add_argument("--runs", nargs=2, required=True, help="A y B: [alias=]retriever/variante")
    p.add_argument("--variant", required=True, help="variante con la que se muestra el texto de la consulta")
    p.add_argument("--domains", nargs="*", default=DOMAINS)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--chars", type=int, default=300, help="caracteres del documento que se muestran")
    p.add_argument("--name", default="runs", help="sufijo del archivo de salida")
    args = parser.parse_args()
    {"expansiones": cmd_expansiones, "anios": cmd_anios, "grupos": cmd_grupos, "ejemplos": cmd_ejemplos}[args.cmd](args)


if __name__ == "__main__":
    main()
