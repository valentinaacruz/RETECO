"""Fusiona runs ya calculados con RRF y registra el resultado (p. ej. R5: híbrido BM25 + denso).

Uso:
    .venv/bin/python scripts/fuse_runs.py --subtrack 1a --runs bm25/limpia bge-base/limpia
    .venv/bin/python scripts/fuse_runs.py --subtrack 1b --runs bge-base/paso bge-base/limpia --weights 1 0.5

El run fusionado se guarda como ``fusion/<nombre>``; por defecto el nombre es
``rrf(<run1>+<run2>...)``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.data import DOMAINS  # noqa: E402
from reteco.evaluate import evaluate_variant, has_run, load_run, log_result, save_run  # noqa: E402
from reteco.fusion import rrf_fuse  # noqa: E402

DEFAULT_REF = {"1a": "bm25/cruda", "1b": "bm25/oficial"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="runs retriever/variante")
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--k", type=int, default=60)
    parser.add_argument("--name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--ref", default=None)
    args = parser.parse_args()

    name = args.name or "rrf(" + "+".join(args.runs).replace("/", ":") + ")"
    if args.weights:
        name += "w" + ",".join(f"{w:g}" for w in args.weights)
    parts = [r.split("/", 1) for r in args.runs]
    domains = [d for d in DOMAINS if all(has_run(r, v, d, args.split) for r, v in parts)]
    for dom in domains:
        runs = [load_run(r, v, dom, args.split) for r, v in parts]
        save_run(rrf_fuse(runs, k=args.k, weights=args.weights), "fusion", name, dom, args.split)

    ref_name = args.ref or DEFAULT_REF[args.subtrack]
    ref = evaluate_variant(*ref_name.split("/", 1), args.subtrack, args.split, domains)
    pq = evaluate_variant("fusion", name, args.subtrack, args.split, domains)
    row = log_result(args.subtrack, "fusion", name, pq, args.split, False, ref_name, ref)
    print(f"{name}: {len(domains)} dominios · nDCG@10={row['ndcg10']:.4f} R@100={row['r100']:.4f} "
          f"R@1000={row['r1000']:.4f} · Δ vs {ref_name}={row.get('delta_ndcg10', float('nan')):+.4f} "
          f"p={row.get('p', float('nan'))} dominios +{row.get('dom_mejora')}/-{row.get('dom_empeora')}")


if __name__ == "__main__":
    main()
