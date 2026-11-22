"""O4: ¿merece la pena clasificar la consulta para elegir la estrategia de búsqueda?

Compara, sobre runs ya calculados (solo variantes realistas, no oráculos):

1. **Mejor estrategia única**: la variante con mayor macro nDCG@10 para todas las consultas.
2. **Enrutamiento por grupo (CV)**: para cada clase (o intención) real de ``guidance``,
   se elige la mejor variante en 4/5 de las consultas y se aplica al 1/5 restante.
   Es el techo de un clasificador perfecto, sin sobreajustar la elección. Con
   ``--group domain`` el grupo es el dominio, que sí se conoce en test: es un
   enrutamiento realista, sin clasificador.
3. **Enrutamiento por grupo (en muestra)**: lo mismo eligiendo con todas las consultas (optimista).
4. **Oráculo por consulta**: la mejor variante de cada consulta. Cota superior de cualquier enrutamiento.

Si (2) no supera a (1) de forma significativa, el clasificador no aporta (Fase 3 del plan).

Los candidatos pueden ser retrievers distintos (enrutamiento "duro") o la misma fusión
con distintos pesos (enrutamiento "suave": elegir los pesos de RRF por grupo).

En 1b los folds se agrupan por consulta madre (``StratifiedGroupKFold``): los pasos de
una misma consulta comparten etiqueta y se parecen entre sí, así que separarlos entre
entrenamiento y prueba haría la elección algo optimista. En 1a cada consulta es su propio
grupo y se usa ``StratifiedKFold``, como en la primera versión de O4.

Uso:
    .venv/bin/python scripts/oracle_routing.py --subtrack 1a --candidates bm25/limpia bm25/titulo bge-base/limpia

Argumentos: ``--candidates`` (runs ya calculados; el primero fija el orden de las
consultas), ``--group`` (``clase``, ``intent`` o ``domain``), ``--split``, ``--folds`` y
``--tag`` (sufijo del archivo de salida, para distinguir conjuntos de candidatos).

Salida: imprime la comparación y guarda la tabla por grupo en
``outputs/exp/oracles/O4_routing_<subtrack>_<grupo>[_<tag>].csv``.
"""

import argparse
import sys
from pathlib import Path

# Permite importar el paquete reteco (en src/) sin instalarlo; por eso los imports que
# siguen llevan "noqa: E402" (import que no está al principio del archivo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402

from reteco.data import EXP_DIR, SEED, parent_query_id  # noqa: E402
from reteco.evaluate import evaluate_variant, paired_test  # noqa: E402

METRIC = "ndcg_cut_10"


def macro_of(values: pd.Series, domains: pd.Series) -> float:
    """Macro (media de medias por dominio) de un valor por consulta.

    Args:
        values: métrica por consulta.
        domains: dominio de cada consulta (mismo índice).
    """
    return float(values.groupby(domains).mean().mean())


def as_frame(values: pd.Series, domains: pd.Series) -> pd.DataFrame:
    """Formato que espera ``paired_test``: columnas ``ndcg_cut_10`` y ``domain``."""
    return pd.DataFrame({METRIC: values, "domain": domains})


def choose(M: pd.DataFrame, groups: pd.Series, domains: pd.Series, fallback: str) -> dict[str, str]:
    """Mejor candidato por grupo (media por consulta dentro del grupo); grupos de < 10 consultas usan ``fallback``.

    Args:
        M: métrica por consulta (filas) y candidato (columnas).
        groups: grupo (clase o intención) de cada consulta.
        domains: dominio de cada consulta (no se usa; se mantiene por simetría con ``macro_of``).
        fallback: candidato para los grupos pequeños, donde elegir sería ruido.

    Returns:
        ``{grupo: candidato elegido}``.
    """
    out = {}
    for g, idx in M.groupby(groups).groups.items():
        out[g] = M.loc[idx].mean().idxmax() if len(idx) >= 10 else fallback
    return out


def main() -> None:
    """Calcula las cuatro estrategias de O4 e imprime la comparación."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--candidates", nargs="+", required=True, help="runs retriever/variante")
    parser.add_argument("--group", choices=["clase", "intent", "domain"], default="clase")
    parser.add_argument("--split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tag", default="", help="sufijo del CSV de salida")
    args = parser.parse_args()

    frames = {}
    for cand in args.candidates:
        retriever, variant = cand.split("/", 1)
        frames[cand] = evaluate_variant(retriever, variant, args.subtrack, args.split)
    # Matriz consultas × candidatos. La clave "dominio|qid" es única entre dominios; se
    # asume que todos los candidatos tienen las mismas consultas en el mismo orden.
    base = frames[args.candidates[0]]
    key = base["domain"] + "|" + base.index.to_series()
    M = pd.DataFrame({c: f[METRIC].to_numpy() for c, f in frames.items()}, index=key)
    domains = pd.Series(base["domain"].to_numpy(), index=key)
    groups = pd.Series(base[args.group].to_numpy(), index=key)

    # (1) Mejor estrategia única.
    single = {c: macro_of(M[c], domains) for c in M}
    best = max(single, key=single.get)

    # (2) Enrutamiento con validación cruzada (estratificada por grupo): en cada fold se
    # elige la regla con las consultas de entrenamiento y se aplica a las de prueba.
    routed_cv = pd.Series(np.nan, index=key)
    if args.subtrack == "1b":
        # Folds por consulta madre: todos los pasos de una consulta caen en el mismo fold.
        parents = [parent_query_id(q) for q in base.index]
        splits = StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED).split(M, groups, parents)
    else:
        splits = StratifiedKFold(args.folds, shuffle=True, random_state=SEED).split(M, groups)
    for tr, te in splits:
        best_tr = max(M.columns, key=lambda c: macro_of(M[c].iloc[tr], domains.iloc[tr]))
        rule = choose(M.iloc[tr], groups.iloc[tr], domains.iloc[tr], best_tr)
        for i in te:
            routed_cv.iloc[i] = M.iloc[i][rule.get(groups.iloc[i], best_tr)]
    # (3) Enrutamiento eligiendo con todas las consultas (en muestra) y (4) oráculo por consulta.
    rule_all = choose(M, groups, domains, best)
    routed_in = pd.Series([M.loc[k, rule_all.get(groups[k], best)] for k in key], index=key)
    per_query_oracle = M.max(axis=1)

    print(f"{args.subtrack} · grupos = {args.group} · {len(M)} consultas · split={args.split}\n")
    print("Estrategias únicas (macro nDCG@10):")
    for c, v in sorted(single.items(), key=lambda x: -x[1]):
        print(f"  {c:40s} {v:.4f}")
    t = paired_test(as_frame(routed_cv, domains), as_frame(M[best], domains))
    print(f"\nMejor única ({best}):              {single[best]:.4f}")
    print(f"Enrutamiento por {args.group} (CV):        {macro_of(routed_cv, domains):.4f}  "
          f"Δ={t['delta']:+.4f}  p={t['p']:.4f}  dominios +{t['mejora']}/-{t['empeora']}")
    print(f"Enrutamiento por {args.group} (en muestra): {macro_of(routed_in, domains):.4f}")
    print(f"Oráculo por consulta (cota superior): {macro_of(per_query_oracle, domains):.4f}")

    # Tabla por grupo: media de cada candidato, número de consultas y variante elegida.
    table = M.groupby(groups).mean().round(4)
    table["n"] = groups.value_counts()
    table["elegida"] = pd.Series(rule_all)
    print(f"\nnDCG@10 medio por {args.group} y variante (micro dentro del grupo):")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(table.to_string())

    suffix = f"_{args.tag}" if args.tag else ""
    out = EXP_DIR / "oracles" / f"O4_routing_{args.subtrack}_{args.group}{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)


if __name__ == "__main__":
    main()
