"""TG0: ¿puede ayudar el filtro temporal de TG-RAG? Cobertura temporal de consultas, pool y gold.

Uso:
    .venv/bin/python scripts/tg_diagnose.py --subtrack 1a --pool p100 --extractor rules --qtime rules
    .venv/bin/python scripts/tg_diagnose.py --subtrack 1a --pool p100 --extractor rules --qtime oracle

Por dominio (y en macro):
    - ``con_alcance``: fracción de consultas con ``T_q`` no vacío.
    - ``recall_pool``: fracción de los gold que están en el pool (techo de cualquier reranker).
    - ``doc_fechado``: fracción de documentos del pool con al menos un hecho con τ.
    - ``gold_en_Tq`` / ``nogold_en_Tq``: entre las consultas con alcance, fracción de los
      documentos gold (y no gold) del pool con algún hecho cuyo τ cae en ``T_q``, para cada
      modo de emparejamiento. ``lift`` es su cociente: si es ≈ 1, el filtro no separa gold
      de no-gold y TG-RAG solo puede ayudar por la parte semántica o de grafo.

Usa qrels de train: es un análisis, no un sistema. Guarda ``outputs/exp/tgrag/diag/tg0_*.csv``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from reteco.data import DOMAINS, load_qrels  # noqa: E402
from reteco.temporal import temporal_match  # noqa: E402
from reteco.tgraph import TG_DIR, load_facts, load_pool, load_qtime, pool_path  # noqa: E402

MODES = {"contained": ("contained", 1.0), "overlap": ("overlap", 1.0), "soft>=0.5": ("soft", 0.5)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--extractor", default="rules")
    parser.add_argument("--qtime", default="rules")
    parser.add_argument("--split", default="train")
    parser.add_argument("--domains", nargs="*", default=None)
    args = parser.parse_args()

    rows = []
    for dom in args.domains or DOMAINS:
        if not pool_path(args.subtrack, args.pool, dom, args.split).exists():
            continue
        pool = load_pool(args.subtrack, args.pool, dom, args.split)
        qrels = load_qrels(dom, args.split, args.subtrack)
        scopes = load_qtime(args.qtime, args.subtrack, dom, args.split)
        docs = {d for c in pool.values() for d in c}
        facts = load_facts(args.extractor, dom, docs)
        times = {d: [f.time for f in fs if f.time is not None] for d, fs in facts.items()}

        row = {"domain": dom, "consultas": len(pool)}
        row["con_alcance"] = sum(bool(scopes.get(q)) for q in pool) / max(len(pool), 1)
        rec = [len(set(c) & {d for d, r in qrels.get(q, {}).items() if r > 0}) / max(1, sum(r > 0 for r in qrels.get(q, {}).values()))
               for q, c in pool.items() if q in qrels]
        row["recall_pool"] = sum(rec) / max(len(rec), 1)
        row["doc_fechado"] = sum(bool(times.get(d)) for d in docs) / max(len(docs), 1)
        for label, (mode, thr) in MODES.items():
            g_hit = g_n = n_hit = n_n = 0
            for q, cands in pool.items():
                sc = scopes.get(q, [])
                if not sc:
                    continue
                gold = {d for d, r in qrels.get(q, {}).items() if r > 0}
                for d in cands:
                    hit = any(temporal_match(t, sc, mode) >= thr for t in times.get(d, []))
                    if d in gold:
                        g_hit, g_n = g_hit + hit, g_n + 1
                    else:
                        n_hit, n_n = n_hit + hit, n_n + 1
            g, n = g_hit / max(g_n, 1), n_hit / max(n_n, 1)
            row[f"gold_en_Tq[{label}]"] = g
            row[f"nogold_en_Tq[{label}]"] = n
            row[f"lift[{label}]"] = g / n if n > 0 else float("nan")
        rows.append(row)
        print(f"{dom}: alcance {row['con_alcance']:.2f} · recall pool {row['recall_pool']:.2f} · "
              f"fechados {row['doc_fechado']:.2f} · lift contained {row['lift[contained]']:.2f}", flush=True)

    df = pd.DataFrame(rows).set_index("domain")
    df.loc["MACRO"] = df.mean(numeric_only=True)
    out = TG_DIR / "diag" / f"tg0_{args.subtrack}_{args.pool}_{args.extractor}_qt-{args.qtime}_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(df)
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
