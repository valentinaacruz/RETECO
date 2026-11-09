"""O4: ¿merece la pena clasificar la consulta para elegir la estrategia de búsqueda?

Compara, sobre runs ya calculados (solo variantes realistas, no oráculos):

1. **Mejor estrategia única**: la variante con mayor macro nDCG@10 para todas las consultas.
2. **Enrutamiento por grupo (CV)**: para cada clase (o intención) real de ``guidance``,
   se elige la mejor variante en 4/5 de las consultas y se aplica al 1/5 restante.
   Es el techo de un clasificador perfecto, sin sobreajustar la elección.
3. **Enrutamiento por grupo (en muestra)**: lo mismo eligiendo con todas las consultas (optimista).
4. **Oráculo por consulta**: la mejor variante de cada consulta. Cota superior de cualquier enrutamiento.

Si (2) no supera a (1) de forma significativa, el clasificador no aporta (Fase 3 del plan).

Uso:
    .venv/bin/python scripts/oracle_routing.py --subtrack 1a --candidates bm25/limpia bm25/titulo bge-base/limpia
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from reteco.data import EXP_DIR, SEED  # noqa: E402
from reteco.evaluate import evaluate_variant, paired_test  # noqa: E402

METRIC = "ndcg_cut_10"


def macro_of(values: pd.Series, domains: pd.Series) -> float:
    return float(values.groupby(domains).mean().mean())


def as_frame(values: pd.Series, domains: pd.Series) -> pd.DataFrame:
    """Formato que espera ``paired_test``: columnas ``ndcg_cut_10`` y ``domain``."""
    return pd.DataFrame({METRIC: values, "domain": domains})


def choose(M: pd.DataFrame, groups: pd.Series, domains: pd.Series, fallback: str) -> dict[str, str]:
    """Mejor candidato por grupo (media por consulta dentro del grupo); grupos de < 10 consultas usan ``fallback``."""
    out = {}
    for g, idx in M.groupby(groups).groups.items():
        out[g] = M.loc[idx].mean().idxmax() if len(idx) >= 10 else fallback
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--candidates", nargs="+", required=True, help="runs retriever/variante")
    parser.add_argument("--group", choices=["clase", "intent"], default="clase")
    parser.add_argument("--split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    frames = {}
    for cand in args.candidates:
        retriever, variant = cand.split("/", 1)
        frames[cand] = evaluate_variant(retriever, variant, args.subtrack, args.split)
    base = frames[args.candidates[0]]
    key = base["domain"] + "|" + base.index.to_series()
    M = pd.DataFrame({c: f[METRIC].to_numpy() for c, f in frames.items()}, index=key)
    domains = pd.Series(base["domain"].to_numpy(), index=key)
    groups = pd.Series(base[args.group].to_numpy(), index=key)

    single = {c: macro_of(M[c], domains) for c in M}
    best = max(single, key=single.get)

    # Enrutamiento con validación cruzada (estratificada por grupo).
    routed_cv = pd.Series(np.nan, index=key)
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
    for tr, te in skf.split(M, groups):
        best_tr = max(M.columns, key=lambda c: macro_of(M[c].iloc[tr], domains.iloc[tr]))
        rule = choose(M.iloc[tr], groups.iloc[tr], domains.iloc[tr], best_tr)
        for i in te:
            routed_cv.iloc[i] = M.iloc[i][rule.get(groups.iloc[i], best_tr)]
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

    table = M.groupby(groups).mean().round(4)
    table["n"] = groups.value_counts()
    table["elegida"] = pd.Series(rule_all)
    print(f"\nnDCG@10 medio por {args.group} y variante (micro dentro del grupo):")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(table.to_string())

    out = EXP_DIR / "oracles" / f"O4_routing_{args.subtrack}_{args.group}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)


if __name__ == "__main__":
    main()
