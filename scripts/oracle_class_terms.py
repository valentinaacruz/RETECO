"""O5: ¿hay términos de documento característicos de cada clase de razonamiento?

Para cada clase primaria compara los documentos gold de sus consultas con los
*hard negatives* de esas mismas consultas (top-50 de BM25 que no son gold) mediante
log-odds con prior de Dirichlet informativo (Monroe et al., 2008). Se cuenta la
frecuencia de documento (presencia/ausencia), no el tf, para que un documento largo
no domine.

Los negativos se limitan a documentos de ID corto: los de ID largo nunca son gold
(EDA 2.4) y tienen un estilo propio (bibliografías, URLs) que sesgaría los términos.
Esto es solo para el análisis; el sistema nunca mira el formato del ID.

Además comprueba las pistas de documento escritas a mano en PLAN_EXPERIMENTOS.md §6.1.

Uso:
    .venv/bin/python scripts/oracle_class_terms.py [--run bm25/limpia] [--neg-depth 50]
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

from reteco.data import DOMAINS, EXP_DIR, iter_documents, load_qrels, query_labels  # noqa: E402
from reteco.evaluate import load_run  # noqa: E402

WORD_RE = re.compile(r"[a-z][a-z'-]{2,}")
SHORT_ID_RE = re.compile(r"^[^/]+/[0-9a-f]{8}_\d+\.txt$")

# Pistas de documento de PLAN_EXPERIMENTOS.md §6.1 (hipótesis a contrastar).
HANDWRITTEN = {
    "event_analysis_and_localization": ["date", "occurred", "happened", "took", "place", "announced", "released"],
    "time_period_contextualization": ["century", "era", "period", "contemporary", "during", "age", "times"],
    "origins_evolution_comparative_analysis": ["originated", "origin", "first", "earliest", "derived", "introduced", "evolved"],
    "trends_changes_and_cross_period": ["increased", "declined", "decline", "growth", "since", "trend", "changed"],
    "event_verification_and_authenticity": ["according", "evidence", "records", "hoax", "disputed", "claim", "myth"],
    "materials_artifacts_and_provenance": ["manufactured", "hallmark", "circa", "material", "maker", "made", "museum"],
}


def doc_terms(text: str, max_words: int = 150) -> set[str]:
    """Términos de las ``max_words`` primeras palabras.

    Truncar iguala la longitud de gold y negativos: los hard negatives de BM25 son
    más largos y, sin truncar, contienen casi cualquier palabra común.
    """
    words = WORD_RE.findall(text.lower())[:max_words]
    return {w for w in words if w not in ENGLISH_STOP_WORDS}


def log_odds_z(c1: Counter, n1: int, c2: Counter, n2: int, prior: Counter, n_prior: int, a0: float = 500.0,
               keep: set[str] | None = None) -> pd.Series:
    """z-score del log-odds ratio (grupo 1 vs grupo 2) con prior de Dirichlet informativo.

    ``a0`` es la masa total del prior: reparte ``a0`` pseudo-documentos según la
    frecuencia de cada término en el fondo, lo que encoge los términos raros.
    """
    terms = [t for t in prior if c1[t] + c2[t] >= 5 and (keep is None or t in keep)]
    a = np.array([a0 * prior[t] / n_prior for t in terms])
    y1 = np.array([c1[t] for t in terms], dtype=float)
    y2 = np.array([c2[t] for t in terms], dtype=float)
    l1 = np.log((y1 + a) / (n1 + a0 - y1 - a))
    l2 = np.log((y2 + a) / (n2 + a0 - y2 - a))
    var = 1 / (y1 + a) + 1 / (y2 + a)
    return pd.Series((l1 - l2) / np.sqrt(var), index=terms).sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="bm25/limpia", help="retriever/variante de donde salen los hard negatives")
    parser.add_argument("--neg-depth", type=int, default=50)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    retriever, variant = args.run.split("/", 1)

    # 1) Qué documentos son gold y cuáles hard negatives, por clase.
    gold_of, neg_of = defaultdict(set), defaultdict(set)   # clase -> {(dominio, doc_id)}
    gold_queries = {}                                       # (dominio, doc_id) gold -> (dominio, qid)
    for dom in DOMAINS:
        labels, qrels = query_labels(dom, args.split), load_qrels(dom, args.split, "1a")
        run = load_run(retriever, variant, dom, args.split)
        for qid, gold in qrels.items():
            clase = labels.get(qid, {}).get("clase", "other")
            gold_of[clase].update((dom, d) for d in gold)
            gold_queries.update({(dom, d): (dom, qid) for d in gold})
            ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])[:args.neg_depth]
            neg_of[clase].update((dom, d) for d, _ in ranked if d not in gold and SHORT_ID_RE.match(d))

    # 2) Términos de cada documento necesario (una pasada por corpus).
    needed = defaultdict(set)
    for groups in (gold_of, neg_of):
        for docs in groups.values():
            for dom, d in docs:
                needed[dom].add(d)
    terms = {}
    for dom in DOMAINS:
        for doc in iter_documents(dom):
            if doc["id"] in needed[dom]:
                terms[(dom, doc["id"])] = doc_terms(doc["content"])
        print(f"{dom}: {len(needed[dom]):,} documentos leídos", flush=True)

    def counts(docs: set) -> Counter:
        c = Counter()
        for key in docs:
            c.update(terms[key])
        return c

    prior = counts(set(terms))
    # Solo términos presentes en gold de ≥ 5 consultas distintas: si no, el top es vocabulario
    # del tema de una única consulta (nombres propios), no un patrón de la clase.
    support = defaultdict(set)
    for key, q in gold_queries.items():
        for t in terms[key]:
            support[t].add(q)
    keep = {t for t, qs in support.items() if len(qs) >= 5}
    all_gold, all_neg = set().union(*gold_of.values()), set().union(*neg_of.values())
    z_all = log_odds_z(counts(all_gold), len(all_gold), counts(all_neg), len(all_neg), prior, len(terms), keep=keep)
    out_rows, hw_rows = [], []
    for clase in sorted(gold_of):
        g, n = gold_of[clase], neg_of[clase]
        z = log_odds_z(counts(g), len(g), counts(n), len(n), prior, len(terms), keep=keep)
        # Específicos de la clase: significativos en ella (z > 2) y bastante más que en el conjunto.
        specific = (z[z > 2] - z_all.reindex(z[z > 2].index).fillna(0)).sort_values(ascending=False).head(20)
        out_rows.append({"clase": clase, "n_gold": len(g), "n_neg": len(n),
                         "top_gold": ", ".join(f"{t} ({v:.1f})" for t, v in z.head(25).items()),
                         "especificos": ", ".join(f"{t} (+{v:.1f})" for t, v in specific.items()),
                         "top_neg": ", ".join(f"{t} ({v:.1f})" for t, v in z.tail(15)[::-1].items())})
        for t in HANDWRITTEN.get(clase, []):
            hw_rows.append({"clase": clase, "termino": t, "z": round(float(z.get(t, np.nan)), 2),
                            "z_global": round(float(z_all.get(t, np.nan)), 2)})

    out_dir = EXP_DIR / "oracles"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_dir / "O5_terminos_por_clase.csv", index=False)
    hw = pd.DataFrame(hw_rows)
    hw.to_csv(out_dir / "O5_diccionario_manual.csv", index=False)
    pd.set_option("display.max_colwidth", 400)
    pd.set_option("display.width", 250)
    print(f"== GLOBAL\n  gold: {', '.join(f'{t} ({v:.1f})' for t, v in z_all.head(25).items())}")
    for r in out_rows:
        print(f"\n== {r['clase']} (gold={r['n_gold']}, neg={r['n_neg']})\n  gold: {r['top_gold']}"
              f"\n  específicos: {r['especificos']}\n  neg:  {r['top_neg']}")
    print("\nPistas manuales (§6.1): z > 2 = más frecuente en gold que en hard negatives de la clase")
    print(hw.pivot_table(index="termino", columns="clase", values="z").round(1).to_string())


if __name__ == "__main__":
    main()
