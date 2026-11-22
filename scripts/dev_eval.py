"""Evalúa en dev los cuatro sistemas de la tabla principal (comprobación única y final).

Uso:
    .venv/bin/python scripts/dev_eval.py

Antes hay que generar los runs de dev con ``run_experiment.py --split dev`` y
``fuse_runs.py --split dev`` (comandos en RESULTADOS_EXPERIMENTOS.md §2.12).

Para cada subtrack imprime:
    - macro nDCG@10, R@100 y R@1000 de cada sistema;
    - pruebas pareadas: cada sistema frente al baseline, el denso frente a BM25 y el
      híbrido frente al denso (Δ, p y dominios que mejoran / empeoran);
    - nDCG@10 por dominio y número de consultas (1a) o pasos (1b) de cada dominio.

No registra nada en ``resultados.csv``: solo lee los runs ya guardados.
"""

import sys
from pathlib import Path

import pandas as pd

# Permite importar el paquete reteco (en src/) sin instalarlo; por eso el import que
# sigue lleva "noqa: E402" (import que no está al principio del archivo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.evaluate import evaluate_variant, macro, paired_test  # noqa: E402

# Sistemas de cada subtrack, en el orden de la progresión: (nombre corto, retriever, variante).
# Son exactamente los de la tabla principal del reporte, con la configuración fijada en train.
SYSTEMS = {
    "1a": [
        ("base", "bm25", "cruda"),
        ("bm25", "bm25", "rrf(limpia,titulo)"),
        ("denso", "bge-base", "rrf(limpia,titulo)"),
        ("hibrido", "fusion", "rrf(bm25:rrf(limpia,titulo)+bge-base:rrf(limpia,titulo))w0.5,1"),
    ],
    "1b": [
        ("base", "bm25", "oficial"),
        ("bm25", "bm25", "titulo+paso_limpio"),
        ("denso", "bge-base", "limpia+paso"),
        ("hibrido", "fusion", "rrf(bm25:titulo+paso_limpio+bge-base:limpia+paso)w0.5,1"),
    ],
}

# Comparaciones extra, además de "frente al baseline": cada sistema frente al paso anterior.
# (El paso anterior de "bm25" es el propio baseline, así que no necesita una comparación extra.)
PREVIOUS = {"denso": "bm25", "hibrido": "denso"}


def describe_test(label: str, test: dict) -> str:
    """Formatea el resultado de ``paired_test`` como ``| vs <label> Δ p=<p> mejoran/empeoran``.

    Args:
        label: nombre del sistema de referencia.
        test: diccionario devuelto por ``paired_test``.
    """
    return f" | vs {label} {test['delta']:+.4f} p={test['p']:.4f} {test['mejora']}/{test['empeora']}"


def main() -> None:
    """Evalúa los sistemas de cada subtrack en dev e imprime macro, pruebas pareadas y tabla por dominio."""
    for subtrack, systems in SYSTEMS.items():
        # Métricas por consulta de cada sistema en dev.
        per_query = {name: evaluate_variant(retriever, variant, subtrack, "dev")
                     for name, retriever, variant in systems}

        print(f"\n== {subtrack} dev ==")
        for name, df in per_query.items():
            line = (f"{name:8s} n={len(df):4d} nDCG={macro(df):.4f} "
                    f"R100={macro(df, 'recall_100'):.3f} R1000={macro(df, 'recall_1000'):.3f}")
            # Toda mejora se mide primero frente al baseline oficial...
            if name != "base":
                line += describe_test("base", paired_test(df, per_query["base"]))
            # ...y luego frente al paso anterior de la progresión.
            if name in PREVIOUS:
                prev = PREVIOUS[name]
                line += describe_test(prev, paired_test(df, per_query[prev]))
            print(line)

        # nDCG@10 medio por dominio de cada sistema, más el número de consultas o pasos.
        table = pd.DataFrame({name: df.groupby("domain")["ndcg_cut_10"].mean()
                              for name, df in per_query.items()})
        table["n"] = per_query["base"].groupby("domain").size()
        print(table.round(4).to_string())


if __name__ == "__main__":
    main()
