"""K4: compara runs con y sin las formas gruesas de las fechas, separando las consultas con fecha.

K4 (``reteco.temporal.coarse_forms``) solo cambia las consultas que tienen alguna fecha;
las demás quedan idénticas a la base. Por eso, además de la macro de todo train, este
script informa del subconjunto **con fecha** (donde está todo el efecto) y del Δ por dominio.

Uso:
    .venv/bin/python scripts/k4_diag.py --subtrack 1a \\
        --pairs "bm25/rrf(limpia,titulo)" "bm25/rrf(limpia,titulo)+k4" \\
                "bge-base/rrf(limpia,titulo)" "bge-base/rrf(limpia,titulo)+k4"

Argumentos:
    --subtrack  1a o 1b.
    --pairs     pares ``base nuevo`` de runs ``retriever/variante`` (ya calculados).
    --split     split (train por defecto).
    --coverage  imprime además la cobertura de K4 por dominio y ejemplos de formas.

Salida: una tabla por par con nDCG@10 de la base y del nuevo en todo el split y en el
subconjunto con fecha (macro por dominio), Δ, p (``paired_test``), dominios que mejoran /
empeoran y R@100; y el Δ por dominio en el subconjunto con fecha.
"""

import argparse
import sys
from pathlib import Path

# Permite importar el paquete reteco (en src/) sin instalarlo; por eso los imports que
# siguen llevan "noqa: E402" (import que no está al principio del archivo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from reteco.data import DOMAINS, clean_query, load_examples, load_steps  # noqa: E402
from reteco.evaluate import evaluate_variant, macro, paired_test  # noqa: E402
from reteco.queries import k4_forms  # noqa: E402


def k4_table(subtrack: str, split: str) -> pd.DataFrame:
    """Formas K4 de cada consulta (1a) o paso (1b), con el mismo texto que usan las variantes.

    Returns:
        DataFrame indexado por qid (``step_id`` en 1b) con ``domain`` y ``forms`` (``""`` si no hay).
    """
    rows = []
    for dom in DOMAINS:
        if subtrack == "1a":
            for r in load_examples(dom, split):
                rows.append((r["id"], dom, k4_forms(clean_query(r["query"]))))
        else:
            for r in load_steps(dom, split):
                q = clean_query(r["query"])
                for s in r["steps"]:
                    rows.append((s["step_id"], dom, k4_forms(f"{q} {s['step_instruction']}")))
    return pd.DataFrame(rows, columns=["qid", "domain", "forms"]).set_index("qid")


def compare(base: pd.DataFrame, new: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    """nDCG@10 y R@100 macro de base y nuevo (opcionalmente solo en ``mask``), Δ, p y dominios.

    Args:
        base, new: métricas por consulta (``evaluate_variant``).
        mask: serie booleana por qid; si se da, solo se comparan esas consultas.
    """
    if mask is not None:
        keep = mask[mask].index
        base, new = base.loc[base.index.intersection(keep)], new.loc[new.index.intersection(keep)]
    t = paired_test(new, base)
    return {"n": len(new), "base": macro(base), "nuevo": macro(new), "delta": t["delta"], "p": t["p"],
            "dom": f"{t['mejora']}/{t['empeora']}", "r100_base": macro(base, "recall_100"),
            "r100_nuevo": macro(new, "recall_100")}


def main() -> None:
    """Imprime la cobertura (opcional) y la comparación de cada par base → nuevo."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="pares base nuevo (retriever/variante)")
    parser.add_argument("--split", default="train")
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    if len(args.pairs) % 2:
        parser.error("--pairs necesita un número par de runs (base nuevo base nuevo ...)")

    forms = k4_table(args.subtrack, args.split)
    has_date = forms["forms"] != ""

    if args.coverage:
        cov = has_date.groupby(forms["domain"]).agg(["sum", "count", "mean"])
        print(f"Cobertura de K4 ({args.subtrack}, {args.split}): {has_date.sum()}/{len(has_date)} "
              f"= {has_date.mean():.1%} (macro por dominio {cov['mean'].mean():.1%})")
        print(cov.rename(columns={"sum": "con_fecha", "count": "n", "mean": "frac"}).round(3).to_string())
        n_forms = forms.loc[has_date, "forms"].str.count(r"\bcentury\b|\d0s\b")
        print(f"Formas por consulta con fecha: mediana {n_forms.median():.0f}, p90 {n_forms.quantile(0.9):.0f}")
        print("Ejemplos:", forms.loc[has_date, "forms"].sample(8, random_state=0).tolist(), "\n")

    for base_name, new_name in zip(args.pairs[::2], args.pairs[1::2]):
        base = evaluate_variant(*base_name.split("/", 1), args.subtrack, args.split)
        new = evaluate_variant(*new_name.split("/", 1), args.subtrack, args.split)
        print(f"== {new_name}  vs  {base_name}")
        for label, mask in (("todo", None), ("con fecha", has_date), ("sin fecha", ~has_date)):
            c = compare(base, new, mask)
            print(f"  {label:<10} n={c['n']:>5}  base {c['base']:.4f} → {c['nuevo']:.4f}  Δ {c['delta']:+.4f}  "
                  f"p {c['p']:.3f}  +/-dom {c['dom']:<6} R@100 {c['r100_base']:.3f} → {c['r100_nuevo']:.3f}")
        # Δ por dominio en el subconjunto con fecha (media por consulta dentro del dominio).
        keep = has_date[has_date].index
        d = (new.loc[new.index.intersection(keep), "ndcg_cut_10"]
             - base.loc[base.index.intersection(keep), "ndcg_cut_10"])
        per_dom = d.groupby(forms.loc[d.index, "domain"]).agg(["mean", "count"])
        print("  Δ por dominio (con fecha):",
              " · ".join(f"{dom} {row['mean']:+.3f} ({int(row['count'])})" for dom, row in per_dom.iterrows()))


if __name__ == "__main__":
    main()
