"""TG-RAG como reranker: grafo temporal del pool + Algoritmo 1 + fusión con la primera etapa.

Uso:
    .venv/bin/python scripts/tg_retrieve.py --subtrack 1a --pool p100 --extractor rules --qtime rules \
        --base "bm25/rrf(limpia,titulo)" --presets full no_ppr no_temporal no_temporal_ppr static
    .venv/bin/python scripts/tg_retrieve.py --subtrack 1a --pool p100 --extractor rules --qtime oracle \
        --base "bm25/rrf(limpia,titulo)" --presets full --K 20 50 200 --match contained overlap soft
    .venv/bin/python scripts/tg_retrieve.py --subtrack 1b --pool p100 --extractor rules --qtime rules \
        --base bm25/titulo+paso_limpio --query-variant titulo+paso_limpio

Por cada combinación de preset × K × match × graph_scope se guardan:
    - el run TG puro ``tgrag-<extractor>-<encoder>/<config>,qt-<qtime>,<pool>`` (solo
      los documentos con score TG; diagnóstico);
    - un run fusionado por cada ``--rrf-weights w``: RRF(base, TG) con pesos (1, w),
      sufijo ``+rrf<w>``.
Todos se evalúan y se registran en ``resultados.csv`` frente a ``--base``. Con
``--subset mini`` solo se usan y se evalúan las consultas de ``train-mini`` (sufijo ``@mini``).

Usa torch (embeddings); no carga la JVM de BM25: los runs base se leen de disco.
"""

import argparse
import csv
import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.data import DOMAINS, parent_query_id, train_mini_ids  # noqa: E402
from reteco.evaluate import evaluate_variant, has_run, load_run, log_result, save_run  # noqa: E402
from reteco.fusion import rrf_fuse  # noqa: E402
from reteco.queries import VARIANTS  # noqa: E402
from reteco.tgrag import PRESETS, TGConfig, description_embeddings, query_embeddings, score_query  # noqa: E402
from reteco.tgraph import TG_DIR, TemporalGraph, load_facts, load_pool, load_qtime  # noqa: E402

DEFAULT_QUERY_VARIANT = {"1a": "limpia", "1b": "titulo+paso_limpio"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--extractor", default="rules", help="carpeta de hechos: rules, rules-inherit, llm-<modelo>")
    parser.add_argument("--qtime", default="rules", help="método de T_q: rules, rules-present, llm, oracle")
    parser.add_argument("--base", required=True, help="run de primera etapa retriever/variante")
    parser.add_argument("--encoder", default="bge-small", help="encoder de reteco.dense para aristas y consultas")
    parser.add_argument("--query-variant", default=None, help="variante de reteco.queries para e_q")
    parser.add_argument("--presets", nargs="+", default=["full"], choices=list(PRESETS))
    parser.add_argument("--K", nargs="+", type=int, default=[20])
    parser.add_argument("--match", nargs="+", default=["contained"], choices=["contained", "overlap", "soft"])
    parser.add_argument("--graph-scope", nargs="+", default=["subgraph"], choices=["subgraph", "pool"])
    parser.add_argument("--edge-scope", default="query", choices=["query", "pool"])
    parser.add_argument("--chunk-weight", default="prod", choices=["prod", "none"])
    parser.add_argument("--doc-agg", default="max", choices=["max", "sum"])
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--top", type=int, default=None, help="solo los N primeros candidatos del pool")
    parser.add_argument("--rrf-weights", nargs="*", type=float, default=[0.5, 1.0])
    parser.add_argument("--subset", choices=["mini"], default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--domains", nargs="*", default=None)
    args = parser.parse_args()

    base_r, base_v = args.base.split("/", 1)
    qvar = args.query_variant or DEFAULT_QUERY_VARIANT[args.subtrack]
    retriever = f"tgrag-{args.extractor}-{args.encoder}"
    suffix = f",qt-{args.qtime},{args.pool}" + (f",top{args.top}" if args.top else "") + \
             (f",q-{qvar}" if args.query_variant else "") + ("@mini" if args.subset else "")
    oracle = args.qtime == "oracle" or VARIANTS[args.subtrack][qvar].oracle

    configs = []
    for preset, k, match, gscope in itertools.product(args.presets, args.K, args.match, args.graph_scope):
        # Sin componente temporal, match no influye: se evita repetir la misma config.
        if not PRESETS[preset]["temporal"] and match != args.match[0]:
            continue
        cfg = replace(TGConfig(), **PRESETS[preset], K=k, match=match, graph_scope=gscope, edge_scope=args.edge_scope,
                      chunk_weight=args.chunk_weight, doc_agg=args.doc_agg, fallback=not args.no_fallback)
        configs.append((cfg.name(preset) + suffix, cfg))

    domains = [d for d in (args.domains or DOMAINS) if has_run(base_r, base_v, d, args.split)]
    mini_all: set[str] = set()
    diag_rows = []
    for dom in domains:
        t0 = time.time()
        pool = load_pool(args.subtrack, args.pool, dom, args.split)
        mini = train_mini_ids(dom, args.split) if args.subset == "mini" else None
        if mini is not None:
            mini_all |= mini
            pool = {q: c for q, c in pool.items() if parent_query_id(q) in mini}
        if args.top:
            pool = {q: c[:args.top] for q, c in pool.items()}
        docs = {d for c in pool.values() for d in c}
        facts = load_facts(args.extractor, dom, docs)
        missing = len(docs) - len(facts)
        if missing:
            print(f"  aviso: {dom}: {missing:,} documentos del pool sin extraer (cuentan como vacíos)")
        graph = TemporalGraph({d: facts.get(d, []) for d in sorted(docs)})
        desc_emb = description_embeddings(graph, args.extractor, args.encoder, dom)
        queries = {q: t for q, t in VARIANTS[args.subtrack][qvar].build(dom, args.split).items() if q in pool}
        qemb = query_embeddings(args.encoder, queries)
        scopes = load_qtime(args.qtime, args.subtrack, dom, args.split)
        base = {q: r for q, r in load_run(base_r, base_v, dom, args.split).items() if q in pool}

        for name, cfg in configs:
            tg_run = {}
            for qid, cands in pool.items():
                cand_idx = [graph.doc_index[d] for d in cands]
                scores, info = score_query(graph, desc_emb, qemb[qid], scopes.get(qid, []), cand_idx, cfg)
                tg_run[qid] = {graph.doc_ids[i]: float(s) for i, s in scores.items()}
                diag_rows.append({"config": name, "domain": dom, "qid": qid, **info})
            save_run(tg_run, retriever, name, dom, args.split)
            for w in args.rrf_weights:
                save_run(rrf_fuse([base, tg_run], weights=[1.0, w]), retriever, f"{name}+rrf{w:g}", dom, args.split)
        print(f"{dom}: {graph.stats()} · {len(configs)} config(s) en {time.time() - t0:.0f} s", flush=True)
        del graph, desc_emb

    # Diagnóstico por consulta (semillas, fallback, documentos con score TG).
    diag_path = TG_DIR / "diag" / f"{retriever}__{args.subtrack}__{args.split}{suffix}.csv"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0]))
        w.writeheader()
        w.writerows(diag_rows)

    # Evaluación y registro frente a la primera etapa (mismas consultas).
    qids = mini_all if args.subset == "mini" else None
    ref = evaluate_variant(base_r, base_v, args.subtrack, args.split, domains, qids)
    print(f"\n{args.subtrack} · {retriever} · referencia={args.base} (nDCG@10 {ref['ndcg_cut_10'].groupby(ref['domain']).mean().mean():.4f})")
    print(f"{'variante':70s} {'nDCG@10':>8s} {'R@100':>7s} {'Δ':>8s} {'p':>7s} {'+/-dom':>7s}")
    for name, _ in configs:
        for variant in [name] + [f"{name}+rrf{w:g}" for w in args.rrf_weights]:
            pq = evaluate_variant(retriever, variant, args.subtrack, args.split, domains, qids)
            row = log_result(args.subtrack, retriever, variant, pq, args.split, oracle, args.base, ref)
            print(f"{variant:70s} {row['ndcg10']:8.4f} {row['r100']:7.4f} {row.get('delta_ndcg10', float('nan')):+8.4f} "
                  f"{row.get('p', float('nan')):7.4f} {row.get('dom_mejora', 0):>3}/{row.get('dom_empeora', 0):<3}")


if __name__ == "__main__":
    main()
