"""Evaluación de runs: métrica oficial, macro por dominio, test pareado y registro de resultados.

Protocolo (PLAN_EXPERIMENTOS.md §1): una variante se acepta frente a su referencia si
sube el macro nDCG@10 ≥ 0.01, el test pareado da p < 0.05 y mejora en ≥ 8 de 13 dominios.
"""

import csv
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytrec_eval

from reteco.data import DOMAINS, EXP_DIR, SEED, load_qrels, parent_query_id, query_labels

MEASURES = ("ndcg_cut.10", "recall.10,100,1000")
METRICS = ["ndcg_cut_10", "recall_10", "recall_100", "recall_1000"]
RESULTS_CSV = EXP_DIR / "resultados.csv"
RUNS_DIR = EXP_DIR / "runs"


# ----------------------------------------------------------------------------- runs en disco


def run_path(retriever: str, variant: str, domain: str, split: str) -> Path:
    """Ruta de un run: ``outputs/exp/runs/<retriever>/<variante>/<dominio>__<split>.pkl``."""
    return RUNS_DIR / retriever / variant / f"{domain}__{split}.pkl"


def save_run(run: dict, retriever: str, variant: str, domain: str, split: str) -> None:
    path = run_path(retriever, variant, domain, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(run, f)


def load_run(retriever: str, variant: str, domain: str, split: str) -> dict:
    with open(run_path(retriever, variant, domain, split), "rb") as f:
        return pickle.load(f)


def has_run(retriever: str, variant: str, domain: str, split: str) -> bool:
    return run_path(retriever, variant, domain, split).exists()


# ----------------------------------------------------------------------------- métricas


def evaluate_run(run: dict, qrels: dict) -> pd.DataFrame:
    """Métricas por consulta con ``pytrec_eval``.

    Las consultas de ``qrels`` que falten en el run cuentan con 0, para que un run
    incompleto no salga favorecido.
    """
    ev = pytrec_eval.RelevanceEvaluator(qrels, set(MEASURES))
    df = pd.DataFrame.from_dict(ev.evaluate(run), orient="index").reindex(list(qrels)).fillna(0.0)
    return df[METRICS]


def evaluate_variant(retriever: str, variant: str, subtrack: str, split: str = "train",
                     domains: list[str] | None = None, qids: set[str] | None = None) -> pd.DataFrame:
    """Métricas por consulta (o paso) en los dominios con run, con ``domain``, ``clase`` e ``intent``.

    Con ``qids`` (ids de consulta madre) solo se evalúan esas consultas, p. ej. ``train-mini``.
    """
    frames = []
    for dom in domains or DOMAINS:
        if not has_run(retriever, variant, dom, split):
            continue
        qrels = load_qrels(dom, split, subtrack)
        if qids is not None:
            qrels = {q: r for q, r in qrels.items() if parent_query_id(q) in qids}
            if not qrels:
                continue
        df = evaluate_run(load_run(retriever, variant, dom, split), qrels)
        labels = query_labels(dom, split)
        parent = [parent_query_id(q) for q in df.index]
        df["clase"] = [labels.get(p, {}).get("clase", "other") for p in parent]
        df["intent"] = [labels.get(p, {}).get("intent", "other") for p in parent]
        frames.append(df.assign(domain=dom))
    return pd.concat(frames) if frames else pd.DataFrame()


def macro(per_query: pd.DataFrame, metric: str = "ndcg_cut_10") -> float:
    """Media por dominio y luego media entre dominios (agregación oficial)."""
    return float(per_query.groupby("domain")[metric].mean().mean())


def paired_test(a: pd.DataFrame, b: pd.DataFrame, metric: str = "ndcg_cut_10",
                n_perm: int = 10_000, seed: int = SEED) -> dict:
    """Compara dos sistemas sobre las mismas consultas.

    Usa un test de aleatorización por signo: bajo la hipótesis nula, intercambiar
    los dos sistemas en cada consulta no cambia nada, así que se invierte el signo
    de cada diferencia al azar y se recalcula la diferencia macro.

    Args:
        a: métricas por consulta del sistema nuevo.
        b: métricas por consulta de la referencia.
        metric: métrica comparada.
        n_perm: número de permutaciones.
        seed: semilla.

    Returns:
        ``{"delta": Δ macro, "p": p-valor bilateral, "mejora": dominios con Δ > 0,
        "empeora": dominios con Δ < 0, "n_dom": dominios comparados}``.
    """
    j = a[[metric, "domain"]].join(b[[metric]], rsuffix="_ref", how="inner")
    diff = (j[metric] - j[f"{metric}_ref"]).to_numpy()
    dom_codes, dom_index = np.unique(j["domain"].to_numpy(), return_inverse=True)
    counts = np.bincount(dom_index)

    def macro_of(d: np.ndarray) -> np.ndarray:
        # d: (n_perm, n_consultas) → macro de cada fila (media de medias por dominio).
        sums = np.zeros((d.shape[0], len(dom_codes)))
        for k in range(len(dom_codes)):
            sums[:, k] = d[:, dom_index == k].sum(axis=1)
        return (sums / counts).mean(axis=1)

    observed = float(macro_of(diff[None, :])[0])
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = macro_of(signs * diff[None, :])
    p = float((np.abs(null) >= abs(observed) - 1e-12).mean())
    by_dom = pd.Series(diff).groupby(j["domain"].to_numpy()).mean()
    return {"delta": observed, "p": p, "mejora": int((by_dom > 1e-9).sum()),
            "empeora": int((by_dom < -1e-9).sum()), "n_dom": len(dom_codes)}


def summary(per_query: pd.DataFrame) -> dict:
    """Macro de cada métrica."""
    return {m: macro(per_query, m) for m in METRICS}


def by_group(per_query: pd.DataFrame, col: str, metric: str = "ndcg_cut_10") -> pd.Series:
    """Media de la métrica por valor de ``col`` (``clase``, ``intent``, ``domain``)."""
    return per_query.groupby(col)[metric].mean()


# ----------------------------------------------------------------------------- registro

CSV_FIELDS = ["fecha", "subtrack", "retriever", "variante", "oraculo", "split", "n_dom",
              "ndcg10", "r10", "r100", "r1000", "referencia", "delta_ndcg10", "p", "dom_mejora", "dom_empeora"]


def log_result(subtrack: str, retriever: str, variant: str, per_query: pd.DataFrame, split: str,
               oracle: bool = False, ref_name: str = "", ref: pd.DataFrame | None = None) -> dict:
    """Añade (o reemplaza) la fila de un experimento en ``outputs/exp/resultados.csv``."""
    s = summary(per_query)
    row = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "subtrack": subtrack, "retriever": retriever,
           "variante": variant, "oraculo": oracle, "split": split, "n_dom": per_query["domain"].nunique(),
           "ndcg10": round(s["ndcg_cut_10"], 4), "r10": round(s["recall_10"], 4),
           "r100": round(s["recall_100"], 4), "r1000": round(s["recall_1000"], 4), "referencia": ref_name}
    if ref is not None and not ref.empty and (ref_name != f"{retriever}/{variant}"):
        t = paired_test(per_query, ref)
        row.update({"delta_ndcg10": round(t["delta"], 4), "p": round(t["p"], 4),
                    "dom_mejora": t["mejora"], "dom_empeora": t["empeora"]})
    rows = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if (r["subtrack"], r["retriever"], r["variante"], r["split"]) != (subtrack, retriever, variant, split)]
    rows.append(row)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return row
