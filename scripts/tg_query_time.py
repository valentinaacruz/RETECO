"""Alcance temporal ``T_q`` de cada consulta (1a) o paso (1b): identificación temporal de TG-RAG (§III-B1).

Uso:
    .venv/bin/python scripts/tg_query_time.py --subtrack 1a --method rules
    .venv/bin/python scripts/tg_query_time.py --subtrack 1a --method oracle
    .venv/bin/python scripts/tg_query_time.py --subtrack 1b --method llm --backend mlx

Métodos:
    ``rules``: ``reteco.temporal.query_scopes`` sobre la consulta limpia.
    ``rules-present``: igual, y además "currently / still / nowadays" → los últimos 10 años.
    ``llm``: prompt ``tg_qtime_v1``, adaptado del *Temporal Extraction Prompt* del apéndice del paper.
    ``oracle``: reglas sobre ``key_time_anchors`` y ``query_temporal_signals`` de guidance.
        **Es un oráculo**: solo sirve para medir el techo de la identificación temporal.

En 1b se usa el alcance del paso; si el paso no tiene, el de su consulta madre.
Guarda ``outputs/exp/tgrag/qtime/<método>/<subtrack>/<dominio>__<split>.json``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.data import DOMAINS, clean_query, load_examples, load_guidance, load_steps, split_query  # noqa: E402
from reteco.queries import first_words  # noqa: E402
from reteco.temporal import Scope, parse_normalized, query_scopes, range_span  # noqa: E402
from reteco.tgraph import qtime_path  # noqa: E402

QTIME_PROMPT_ID = "tg_qtime_v1"
QTIME_PROMPT = """-Goal-
Given a user question that may ask a time-related question, identify every explicit or implicit temporal expression and determine the time scope needed to answer it.
For each time scope output an object with:
- logic: one of at, before, after, between
- start: the normalized time
- end: the normalized end time (only for between)
Normalized time format: YYYY (year; negative for BCE, e.g. -300), YYYY-MM, YYYY-MM-DD, YYYY-Qn, YYY0s (decade, e.g. 1830s), YY00s (century, e.g. 1800s), CNN (century number, e.g. C19; C-5 is the 5th century BCE), present.
If the question has no temporal scope, output [].
Output only a JSON list.

-Examples-
Question: Who was the CEO of DXC Technology on January 1, 2022?
Output: [{"logic": "at", "start": "2022-01-01"}]

Question: What tools were used for mounted cattle herding in cultures prior to 1600?
Output: [{"logic": "before", "start": "1600"}]

Question: How did railway gauges in Britain change between the 1830s and the 1870s?
Output: [{"logic": "between", "start": "1830s", "end": "1870s"}]

Question: Is it still possible to mine Bitcoin profitably with a GPU?
Output: [{"logic": "at", "start": "present"}]

Question: What does a stealth address in Monero do?
Output: []

-Real Data-
Question: {query}
Output:"""


def llm_scopes(llm, item_id: str, text: str) -> list[Scope]:
    from reteco.llm import parse_json_block
    out = llm.generate(QTIME_PROMPT_ID, item_id, QTIME_PROMPT.replace("{query}", text))
    data = parse_json_block(out)
    if isinstance(data, dict):
        data = [data]
    scopes = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        logic = str(item.get("logic", "at")).lower().strip()
        start = parse_normalized(item.get("start"))
        if start is None:
            continue
        if logic == "between":
            end = parse_normalized(item.get("end")) or start
            scopes.append(Scope("at", range_span(min(start.start, end.start), max(start.end, end.end) - 1)))
        elif logic in ("at", "before", "after"):
            scopes.append(Scope(logic, start))
    return list(dict.fromkeys(scopes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subtrack", choices=["1a", "1b"], required=True)
    parser.add_argument("--method", choices=["rules", "rules-present", "llm", "oracle"], required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--backend", choices=["mlx", "openai"], default="mlx")
    parser.add_argument("--model", default=None)
    parser.add_argument("--query-words", type=int, default=300, help="palabras de la consulta que ve el LLM")
    args = parser.parse_args()

    llm = None
    if args.method == "llm":
        from reteco.llm import LLM
        llm = LLM(args.backend, args.model, max_tokens=200)

    def scopes_of(item_id: str, text: str) -> list[Scope]:
        if args.method == "llm":
            return llm_scopes(llm, item_id, text)
        return query_scopes(text, present=args.method == "rules-present")

    def query_text(r: dict) -> str:
        if args.method == "llm":
            title, body = split_query(r["query"])
            return f"{title}\n{first_words(body, args.query_words)}"
        return clean_query(r["query"])

    for split in args.splits:
        for dom in args.domains or DOMAINS:
            guidance = load_guidance(dom, split) if args.method == "oracle" else {}

            def parent_scopes(r: dict) -> list[Scope]:
                if args.method == "oracle":
                    qg = guidance.get(r["id"], {}).get("query_guidance", {})
                    anchors = qg.get("key_time_anchors", []) + qg.get("query_temporal_signals", [])
                    return list(dict.fromkeys(s for a in anchors for s in query_scopes(a)))
                return scopes_of(r["id"], query_text(r))

            out = {}
            if args.subtrack == "1a":
                for r in load_examples(dom, split):
                    out[r["id"]] = parent_scopes(r)
            else:
                for r in load_steps(dom, split):
                    parent = parent_scopes(r)
                    for s in r["steps"]:
                        own = [] if args.method == "oracle" else scopes_of(s["step_id"], s["step_instruction"])
                        out[s["step_id"]] = own or parent
            path = qtime_path(args.method, args.subtrack, dom, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({q: [s.to_str() for s in sc] for q, sc in out.items()}, f)
            with_scope = sum(bool(v) for v in out.values())
            print(f"{dom}/{split}: {with_scope}/{len(out)} con alcance temporal")


if __name__ == "__main__":
    main()
