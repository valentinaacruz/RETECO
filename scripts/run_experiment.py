"""Ejecuta variantes de consulta con un retriever, las evalúa y registra el resultado.

Uso:
    .venv/bin/python scripts/run_experiment.py --subtrack 1a --retriever bm25 --variants cruda limpia
    .venv/bin/python scripts/run_experiment.py --subtrack 1b --retriever bge-base --variants all
    .venv/bin/python scripts/run_experiment.py --subtrack 1a --retriever bge-base@first --variants limpia

Retrievers: ``bm25`` (baseline oficial) o un encoder de ``reteco.dense.MODELS``,
opcionalmente con ``@first`` (score del primer trozo) en lugar del mejor trozo.
Los runs ya calculados se reutilizan; ``--force`` los recalcula.
Cada variante se compara con ``--ref`` (por defecto, el baseline oficial).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.data import DOMAINS  # noqa: E402
from reteco.evaluate import evaluate_variant, has_run, log_result, save_run  # noqa: E402
from reteco.fusion import rrf_multi  # noqa: E402
from reteco.queries import VARIANTS  # noqa: E402

DEFAULT_REF = {"1a": "bm25/cruda", "1b": "bm25/oficial"}


def open_index(retriever: str, domain: str):
    """Carga el índice del retriever; la importación es perezosa para no mezclar JVM y torch."""
    if retriever == "bm25":
        from reteco.bm25 import OfficialStyleBM25
        return OfficialStyleBM25(domain)
    from reteco.dense import DenseIndex
    key, _, agg = retriever.partition("@")
    index = DenseIndex(key, domain)
    index.agg = agg or "max"
    return index


def search(index, queries: dict[str, str | list[str]], k: int = 1000) -> dict:
    """Busca cada texto y fusiona con RRF las consultas que tienen varios."""
    flat = {}
    for qid, q in queries.items():
        for i, text in enumerate(q if isinstance(q, list) else [q]):
            flat[(qid, i)] = text
    texts = {f"{qid}\t{i}": text for (qid, i), text in flat.items()}
    if hasattr(index, "agg"):
        res = index.search_many(texts, k, agg=index.agg)
    else:
        res = index.search_many(texts, k)
    grouped: dict[str, list] = {}
    for key, ranking in res.items():
        grouped.setdefault(key.split("\t")[0], []).append(ranking)
    return {qid: rs[0] if len(rs) == 1 else rrf_multi({qid: rs}, depth=k)[qid] for qid, rs in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--variants", nargs="+", default=["all"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--ref", default=None, help="retriever/variante de referencia")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    registry = VARIANTS[args.subtrack]
    variants = list(registry) if args.variants == ["all"] else args.variants
    unknown = [v for v in variants if v not in registry]
    if unknown:
        parser.error(f"variantes desconocidas: {unknown}. Disponibles: {list(registry)}")
    domains = args.domains or DOMAINS

    # 1) Runs que faltan, dominio a dominio (el índice se carga una vez por dominio).
    for dom in domains:
        missing = [v for v in variants if args.force or not has_run(args.retriever, v, dom, args.split)]
        if not missing:
            continue
        t0 = time.time()
        index = open_index(args.retriever, dom)
        for v in missing:
            run = search(index, registry[v].build(dom, args.split))
            save_run(run, args.retriever, v, dom, args.split)
        print(f"{dom}: {len(missing)} variante(s) en {time.time() - t0:.0f} s", flush=True)
        del index

    # 2) Evaluación y registro.
    ref_name = args.ref or DEFAULT_REF[args.subtrack]
    ref_retriever, ref_variant = ref_name.split("/", 1)
    ref = evaluate_variant(ref_retriever, ref_variant, args.subtrack, args.split, domains)
    print(f"\n{args.subtrack} · {args.retriever} · split={args.split} · referencia={ref_name}")
    print(f"{'variante':32s} {'nDCG@10':>8s} {'R@100':>7s} {'R@1000':>7s} {'Δ':>8s} {'p':>7s} {'+/-dom':>7s}")
    for v in variants:
        pq = evaluate_variant(args.retriever, v, args.subtrack, args.split, domains)
        row = log_result(args.subtrack, args.retriever, v, pq, args.split, registry[v].oracle, ref_name, ref)
        if "delta_ndcg10" in row:
            cmp = f"{row['delta_ndcg10']:+8.4f} {row['p']:7.4f} {row['dom_mejora']:>3d}/{row['dom_empeora']:<3d}"
        else:
            cmp = f"{'—':>8s} {'—':>7s} {'—':>7s}"
        tag = " (oráculo)" if registry[v].oracle else ""
        print(f"{v + tag:32s} {row['ndcg10']:8.4f} {row['r100']:7.4f} {row['r1000']:7.4f} {cmp}")


if __name__ == "__main__":
    main()
