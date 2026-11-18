"""Complementariedad de dos runs: qué gold encuentra cada uno que el otro no (PLAN_SPLADE.md, SP1).

Antes de fusionar dos retrievers conviene saber si se complementan. Si casi todos los
gold que encuentra A también los encuentra B, la fusión tiene poco margen. Este script
no calcula nDCG ni escribe en ``resultados.csv``: solo describe el solapamiento.

Uso:
    .venv/bin/python scripts/compare_runs.py --subtrack 1a --runs "splade-ed/rrf(limpia,titulo)" "bge-base/rrf(limpia,titulo)"
    .venv/bin/python scripts/compare_runs.py --subtrack 1b --runs splade-ed/limpia+paso bge-base/limpia+paso --depth 1000

Argumentos:
    --subtrack  1a o 1b (decide los qrels).
    --runs      dos runs ``retriever/variante`` ya calculados (A y B).
    --depth     profundidad a la que se comparan los gold (100 por defecto).
    --split     split (train por defecto).
    --out       CSV opcional con las medidas por consulta.

Medidas, por consulta y como fracción de sus gold (después, media por dominio y media
de los dominios, como la macro del proyecto):
    R_A, R_B      recall a ``depth`` de cada run;
    R_union       recall de la unión de los dos top-``depth`` (techo de cualquier fusión a esa profundidad);
    solo_A/solo_B gold que están en el top-``depth`` de un run y no en el del otro;
    jacc10        solapamiento (Jaccard) de los dos top-10, sobre documentos (no solo gold).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Permite importar el paquete reteco (en src/) sin instalarlo; por eso los imports que
# siguen llevan "noqa: E402" (import que no está al principio del archivo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.data import DOMAINS, load_qrels  # noqa: E402
from reteco.evaluate import has_run, load_run  # noqa: E402


def ranked(ranking: dict[str, float], depth: int) -> list[str]:
    """Los ``depth`` primeros documentos de un ranking ``{doc_id: score}``.

    Se ordena por score descendente; ``sorted`` es estable, así que en los empates se
    mantiene el orden en que se guardó el run (el del corpus, como en el oficial).

    Args:
        ranking: ``{doc_id: score}``.
        depth: cuántos documentos.

    Returns:
        Lista de ids en orden de ranking.
    """
    return [d for d, _ in sorted(ranking.items(), key=lambda x: -x[1])[:depth]]


def compare_domain(run_a: dict, run_b: dict, qrels: dict, depth: int) -> pd.DataFrame:
    """Medidas de complementariedad por consulta en un dominio.

    Args:
        run_a: run A ``{qid: {doc_id: score}}``.
        run_b: run B.
        qrels: ``{qid: {doc_id: relevancia}}``; solo cuentan las consultas con algún gold.
        depth: profundidad de comparación.

    Returns:
        DataFrame indexado por qid con las columnas descritas en el docstring del módulo.
        Una consulta ausente de un run cuenta como ranking vacío en ese run.
    """
    rows = {}
    for qid, rels in qrels.items():
        gold = {d for d, r in rels.items() if r > 0}
        if not gold:
            continue
        a, b = run_a.get(qid, {}), run_b.get(qid, {})
        top_a, top_b = set(ranked(a, depth)), set(ranked(b, depth))
        a10, b10 = set(ranked(a, 10)), set(ranked(b, 10))
        n = len(gold)
        rows[qid] = {
            "R_A": len(gold & top_a) / n,
            "R_B": len(gold & top_b) / n,
            "R_union": len(gold & (top_a | top_b)) / n,
            "solo_A": len(gold & (top_a - top_b)) / n,
            "solo_B": len(gold & (top_b - top_a)) / n,
            "jacc10": len(a10 & b10) / len(a10 | b10) if a10 | b10 else 0.0,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    """Compara los dos runs en los dominios donde existen ambos e imprime la tabla."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--runs", nargs=2, required=True, metavar=("A", "B"))
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    (ret_a, var_a), (ret_b, var_b) = (r.split("/", 1) for r in args.runs)
    domains = [d for d in DOMAINS if has_run(ret_a, var_a, d, args.split) and has_run(ret_b, var_b, d, args.split)]
    if not domains:
        parser.error("no hay ningún dominio con los dos runs calculados")
    frames = []
    for dom in domains:
        df = compare_domain(load_run(ret_a, var_a, dom, args.split), load_run(ret_b, var_b, dom, args.split),
                            load_qrels(dom, args.split, args.subtrack), args.depth)
        frames.append(df.assign(domain=dom))
    per_query = pd.concat(frames)
    per_dom = per_query.groupby("domain").mean(numeric_only=True)
    per_dom["n"] = per_query.groupby("domain").size()

    print(f"A = {args.runs[0]}\nB = {args.runs[1]}\n{args.subtrack} · split={args.split} · profundidad {args.depth}\n")
    print(per_dom.round(3).to_string())
    # Macro del proyecto: media de las medias por dominio (todos los dominios pesan igual).
    macro = per_dom.drop(columns="n").mean()
    print("\nmacro  " + "  ".join(f"{c}={v:.3f}" for c, v in macro.items()))
    if args.out:
        per_query.to_csv(args.out)
        print(f"por consulta → {args.out}")


if __name__ == "__main__":
    main()
