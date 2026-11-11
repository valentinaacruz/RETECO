"""Extrae hechos temporales de los documentos de uno o varios pools (indexación de TG-RAG).

Uso:
    .venv/bin/python scripts/tg_extract.py --extractor rules --pools 1a/p100 1b/p100
    .venv/bin/python scripts/tg_extract.py --extractor rules-inherit --pools 1a/p100
    .venv/bin/python scripts/tg_extract.py --extractor llm --pools 1a/p100 --subset mini --top 20 \
        --domains iota quant law workplace --backend mlx

Extractores:
    ``rules``: reglas, sin LLM (``reteco.tgraph.extract_rules``).
    ``rules-inherit``: igual, pero las oraciones sin tiempo heredan el último τ del chunk.
    ``llm``: prompt ``tg_extract_v1`` con el backend elegido. Los hechos se guardan como
        ``llm-<modelo>``, y las generaciones quedan además en ``outputs/llm_cache/``.

Con los extractores de reglas, ``--drop-time citation metadata`` no usa como τ los años
de referencias bibliográficas ni las fechas de la página (REVISION_EXTRACCION_TGRAG.md
§3.3). Cada combinación va a su propia carpeta de hechos, p. ej. ``rules-nocit-nometa``,
que es el nombre que hay que pasar a ``--extractor`` en ``tg_diagnose.py`` y
``tg_retrieve.py``.

La extracción es incremental: los documentos ya procesados se saltan. Este script no
importa torch ni la JVM, así que el LLM (mlx) tiene la memoria para él.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm.auto import tqdm  # noqa: E402

from reteco.data import DOMAINS, iter_documents, parent_query_id, train_mini_ids  # noqa: E402
from reteco.temporal import ROLES  # noqa: E402
from reteco.tgraph import append_facts, done_doc_ids, extract_llm, extract_rules, facts_path, load_pool, pool_path  # noqa: E402

_ROLE_SUFFIX = {"citation": "cit", "metadata": "meta"}


def needed_docs(pools: list[str], dom: str, splits: list[str], subset: str | None, top: int | None) -> set[str]:
    """Documentos de los pools pedidos (opcionalmente solo ``train-mini`` y el top-N de cada consulta)."""
    docs = set()
    for spec in pools:
        subtrack, name = spec.split("/", 1)
        for split in splits:
            if not pool_path(subtrack, name, dom, split).exists():
                continue
            mini = train_mini_ids(dom, split) if subset == "mini" else None
            for qid, cands in load_pool(subtrack, name, dom, split).items():
                if mini is None or parent_query_id(qid) in mini:
                    docs.update(cands[:top] if top else cands)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extractor", choices=["rules", "rules-inherit", "llm"], required=True)
    parser.add_argument("--pools", nargs="+", required=True, help="<subtrack>/<pool>, p. ej. 1a/p100")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--subset", choices=["mini"], default=None)
    parser.add_argument("--top", type=int, default=None, help="solo los N primeros candidatos de cada consulta")
    parser.add_argument("--backend", choices=["mlx", "openai"], default="mlx")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-facts", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--drop-time", nargs="*", choices=list(ROLES), default=[],
                        help="papeles de fecha que no se usan como τ (solo extractores de reglas)")
    args = parser.parse_args()

    drop_time = tuple(r for r in ROLES if r in args.drop_time)     # orden fijo → nombre de carpeta estable
    if drop_time and args.extractor == "llm":
        parser.error("--drop-time solo se aplica a los extractores de reglas")
    llm, name = None, args.extractor + "".join(f"-no{_ROLE_SUFFIX[r]}" for r in drop_time)
    if args.extractor == "llm":
        from reteco.llm import LLM
        llm = LLM(args.backend, args.model, max_tokens=args.max_tokens)
        name = f"llm-{llm.model_name.split('/')[-1]}"

    for dom in args.domains or DOMAINS:
        todo = needed_docs(args.pools, dom, args.splits, args.subset, args.top) - done_doc_ids(name, dom)
        if not todo:
            print(f"{dom}: nada que extraer")
            continue
        t0, n_facts = time.time(), 0
        path = facts_path(name, dom)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f, tqdm(total=len(todo), desc=f"{name}:{dom}", leave=False) as bar:
            for doc in iter_documents(dom):
                if doc["id"] not in todo:
                    continue
                if llm is None:
                    facts = extract_rules(doc["content"], inherit_time=args.extractor == "rules-inherit",
                                          drop_time=drop_time)
                else:
                    facts = extract_llm(doc["content"], llm, doc["id"], args.max_facts)
                append_facts(name, dom, doc["id"], facts, f)
                n_facts += len(facts)
                bar.update()
                if llm is not None:
                    f.flush()
        print(f"{dom}: {len(todo):,} documentos, {n_facts:,} hechos en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
