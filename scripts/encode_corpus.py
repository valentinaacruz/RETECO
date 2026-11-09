"""Codifica los corpus del Track 1 con un encoder denso (Fase 1 del plan, R2/R3).

Uso:
    .venv/bin/python scripts/encode_corpus.py --model bge-base
    .venv/bin/python scripts/encode_corpus.py --model e5-base --domains iota law quant

Los dominios ya codificados se saltan, así que se puede interrumpir y relanzar.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.data import DOMAINS, T1_DIR  # noqa: E402
from reteco.dense import MODELS, encode_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="bge-base")
    parser.add_argument("--domains", nargs="*", default=None, help="por defecto, los 13 (de menor a mayor)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # De menor a mayor corpus: los dominios pequeños quedan listos primero para probar.
    domains = args.domains or sorted(DOMAINS, key=lambda d: (T1_DIR / d / "documents.jsonl").stat().st_size)
    for dom in domains:
        encode_corpus(args.model, dom, args.batch_size)


if __name__ == "__main__":
    main()
