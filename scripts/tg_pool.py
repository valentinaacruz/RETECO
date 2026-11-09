"""Pools de candidatos para TG-RAG: unión del top-N de uno o varios runs ya calculados.

Uso:
    .venv/bin/python scripts/tg_pool.py --subtrack 1a --runs "bm25/rrf(limpia,titulo)" --depth 100 --name p100
    .venv/bin/python scripts/tg_pool.py --subtrack 1b --runs bm25/titulo+paso_limpio bge-base/titulo+paso_limpio --name p100

Guarda ``outputs/exp/tgrag/pools/<subtrack>/<nombre>/<dominio>__<split>.json`` con
``{qid: [doc_id, ...]}``. Los documentos se intercalan por rango: primero el top-1 de
cada run, luego el top-2, etc. Solo se procesan los dominios en los que existen todos
los runs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.data import DOMAINS  # noqa: E402
from reteco.evaluate import has_run, load_run  # noqa: E402
from reteco.tgraph import pool_path  # noqa: E402


def ranked(ranking: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(ranking.items(), key=lambda x: -x[1])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="runs retriever/variante")
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--name", required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--domains", nargs="*", default=None)
    args = parser.parse_args()

    parts = [r.split("/", 1) for r in args.runs]
    for split in args.splits:
        for dom in args.domains or DOMAINS:
            if not all(has_run(r, v, dom, split) for r, v in parts):
                print(f"{dom}/{split}: faltan runs, se salta")
                continue
            runs = [load_run(r, v, dom, split) for r, v in parts]
            pool = {}
            for qid in runs[0]:
                lists = [ranked(run.get(qid, {}))[:args.depth] for run in runs]
                seen = {}
                for rank in range(args.depth):
                    for lst in lists:
                        if rank < len(lst):
                            seen.setdefault(lst[rank], None)
                pool[qid] = list(seen)
            path = pool_path(args.subtrack, args.name, dom, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(pool, f)
            n_docs = len({d for ds in pool.values() for d in ds})
            print(f"{dom}/{split}: {len(pool)} consultas, {n_docs:,} documentos únicos")


if __name__ == "__main__":
    main()
