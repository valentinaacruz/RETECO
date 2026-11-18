"""Codifica los corpus del Track 1 con un modelo SPLADE (PLAN_SPLADE.md, SP0–SP2).

Uso (en el servidor; una GPU por proceso, ver PLAN_SPLADE.md §6):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/encode_splade.py --model splade-ed --domains iota quant law
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/encode_splade.py --model splade-v3

Los dominios ya codificados se saltan, así que se puede interrumpir y relanzar. Un
dominio interrumpido a medias se vuelve a codificar entero (el archivo se escribe al final
y de forma atómica).

Argumentos:
    --model       clave de ``reteco.splade.MODELS`` (``splade-ed`` por defecto).
    --domains     dominios a codificar (por defecto los 13, de menor a mayor).
    --batch-size  tamaño de batch del modelo (32; en una 3070 de 8 GB con fp16 caben
                  trozos de 264 tokens; si falta memoria, bajarlo).
    --block       trozos por bloque de codificación (20,000).

Salida: ``outputs/exp/splade/<modelo>/<dominio>.npz`` (ver ``reteco.splade.encode_corpus``).
Por cada dominio imprime el avance, si los trozos coinciden con los de ``bge-base``,
los términos activos por trozo, el tamaño del archivo y el tiempo.
"""

import argparse
import sys
import time
from pathlib import Path

# Permite importar el paquete reteco (en src/) sin instalarlo; por eso los imports que
# siguen llevan "noqa: E402" (import que no está al principio del archivo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.data import DOMAINS, T1_DIR  # noqa: E402
from reteco.splade import BLOCK, DEVICE, MODELS, encode_corpus  # noqa: E402


def main() -> None:
    """Codifica los dominios pedidos, uno detrás de otro."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="splade-ed")
    parser.add_argument("--domains", nargs="*", default=None, help="por defecto, los 13 (de menor a mayor)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block", type=int, default=BLOCK)
    args = parser.parse_args()

    unknown = sorted(set(args.domains or []) - set(DOMAINS))
    if unknown:
        parser.error(f"dominios desconocidos: {unknown}. Disponibles: {DOMAINS}")
    # De menor a mayor corpus: los dominios pequeños quedan listos primero para probar.
    domains = args.domains or sorted(DOMAINS, key=lambda d: (T1_DIR / d / "documents.jsonl").stat().st_size)
    print(f"[{args.model}] {MODELS[args.model]} en {DEVICE} · batch {args.batch_size} · "
          f"dominios: {' '.join(domains)}", flush=True)
    t0 = time.time()
    for dom in domains:
        encode_corpus(args.model, dom, args.batch_size, args.block)
    print(f"[{args.model}] terminado en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
