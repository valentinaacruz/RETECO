"""Revisión de la extracción con reglas de TG-RAG sobre una muestra del corpus (solo lectura).

Uso:
    .venv/bin/python scripts/tg_review_extraction.py                       # 400 docs por dominio, semilla 0
    .venv/bin/python scripts/tg_review_extraction.py --domains history law --show 20

Para cada dominio muestrea documentos (de los ``--pool-size`` primeros, con ``--seed``),
aplica ``reteco.tgraph.extract_rules`` y resume:

- ``hechos/doc`` y ``% con τ``: hechos por documento y fracción con tiempo;
- ``entidades``: entidades distintas; ``% en ≥2 docs``: fracción de menciones cuya entidad
  aparece en al menos dos documentos de la muestra (aproxima la conectividad del grafo,
  que es lo que necesita el PPR);
- ``% con AND`` y ``% >5 pal.``: entidades fusionadas por "and" o demasiado largas;
- ``% cita`` y ``% meta``: fracción de las expresiones temporales con papel ``citation`` o
  ``metadata`` (solo si ``reteco.temporal`` tiene ``extract_spans_with_role``).

Con ``--show N`` imprime N ejemplos al azar de expresiones marcadas como cita y como
metadato, con su contexto, para revisar a mano la precisión. Con ``--show-unflagged N``
imprime N años sueltos *sin* marcar, para estimar cuántas citas se escapan (recall).
Se usó para REVISION_EXTRACCION_TGRAG.md. No escribe nada en ``outputs/``.
"""

import argparse
import itertools
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco import temporal  # noqa: E402
from reteco.data import DOMAINS, iter_documents  # noqa: E402
from reteco.tgraph import _SENT_RE, chunk_text, extract_rules  # noqa: E402

try:
    from reteco.tgraph import is_prose
except ImportError:                     # versión anterior a la revisión: no filtraba oraciones
    def is_prose(_: str) -> bool:
        return True


def sample_docs(domain: str, n: int, pool: int, seed: int) -> list[dict]:
    docs = list(itertools.islice(iter_documents(domain), pool))
    return random.Random(f"{seed}:{domain}").sample(docs, min(n, len(docs)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--docs", type=int, default=400)
    parser.add_argument("--pool-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=0, help="ejemplos de citas y metadatos a imprimir")
    parser.add_argument("--show-unflagged", type=int, default=0, help="años sueltos sin marcar a imprimir")
    args = parser.parse_args()

    with_roles = hasattr(temporal, "extract_spans_with_role")
    examples: dict[str, list[str]] = defaultdict(list)
    cols = ["dominio", "hechos/doc", "% con τ", "entidades", "% en ≥2 docs", "% con AND", "% >5 pal.", "% cita", "% meta"]
    print(" | ".join(cols))
    print(" | ".join("---" for _ in cols))
    totals = Counter()
    for dom in args.domains or DOMAINS:
        docs = sample_docs(dom, args.docs, args.pool_size, args.seed)
        n_facts = n_timed = 0
        ent_docs: dict[str, set[str]] = defaultdict(set)
        mentions = Counter()
        roles = Counter()
        for d in docs:
            facts = extract_rules(d["content"])
            n_facts += len(facts)
            n_timed += sum(f.time is not None for f in facts)
            for f in facts:
                for e in {f.e1, f.e2}:
                    ent_docs[e].add(d["id"])
                    mentions[e] += 1
            if not with_roles:
                continue
            for chunk in chunk_text(d["content"]):
                for sent in _SENT_RE.split(chunk):
                    if not sent or not sent.strip() or not is_prose(sent):
                        continue
                    for s, e, span, role in temporal.extract_spans_with_role(sent):
                        roles[role] += 1
                        ctx = f"[{dom}] {sent[max(0, s - 60):s]}[[{sent[s:e]}]]{sent[e:e + 30]}".replace("\n", " ")
                        if role:
                            examples[role].append(ctx)
                        elif span.gran == "year":
                            examples["unflagged-year"].append(ctx)
        n_ment = sum(mentions.values()) or 1
        shared = sum(c for e, c in mentions.items() if len(ent_docs[e]) >= 2) / n_ment
        with_and = sum(c for e, c in mentions.items() if " AND " in f" {e} ") / n_ment
        long_ = sum(c for e, c in mentions.items() if len(e.split()) > 5) / n_ment
        n_spans = sum(roles.values()) or 1
        cit = f"{roles['citation'] / n_spans:.1%}" if with_roles else "—"
        meta = f"{roles['metadata'] / n_spans:.1%}" if with_roles else "—"
        print(f"{dom} | {n_facts / len(docs):.1f} | {n_timed / max(n_facts, 1):.1%} | {len(mentions):,} | "
              f"{shared:.1%} | {with_and:.2%} | {long_:.2%} | {cit} | {meta}", flush=True)
        totals.update(facts=n_facts, timed=n_timed, docs=len(docs), ments=n_ment,
                      shared=shared * n_ment, cit=roles["citation"], meta=roles["metadata"], spans=sum(roles.values()))
    print(f"**total** | {totals['facts'] / totals['docs']:.1f} | {totals['timed'] / max(totals['facts'], 1):.1%} | | "
          f"{totals['shared'] / totals['ments']:.1%} | | | "
          + (f"{totals['cit'] / max(totals['spans'], 1):.1%} | {totals['meta'] / max(totals['spans'], 1):.1%}"
             if with_roles else "— | —"))

    rng = random.Random(args.seed)
    for role, n in (("citation", args.show), ("metadata", args.show), ("unflagged-year", args.show_unflagged)):
        if n and examples[role]:
            print(f"\n--- {role}: {n} ejemplos al azar de {len(examples[role]):,}")
            for ex in rng.sample(examples[role], min(n, len(examples[role]))):
                print("  " + ex)


if __name__ == "__main__":
    main()
