"""Registro de variantes de consulta (qué texto se usa para buscar).

Cada variante es una función ``(dominio, split) -> {qid: texto | [textos]}``. Si una
consulta tiene varios textos, se busca con cada uno y los rankings se fusionan con
RRF (``scripts/run_experiment.py``).

Las variantes marcadas como oráculo usan ``gold_answers`` o ``guidance``, que no
estarán en test: solo sirven para medir techos (Fase 2 del plan).

Los nombres de las variantes (``cruda``, ``limpia``, ``titulo+paso_limpio``...) son los
que aparecen en ``outputs/exp/runs/`` y en las tablas de RESULTADOS_EXPERIMENTOS.md.
"""

import re
from dataclasses import dataclass
from typing import Callable

from bs4 import BeautifulSoup

from reteco.data import clean_query, load_examples, load_guidance, load_steps, split_query


@dataclass(frozen=True)
class Variant:
    """Una forma de construir el texto de búsqueda.

    Attributes:
        subtrack: ``"1a"`` o ``"1b"``.
        build: función ``(dominio, split) -> {qid: texto | [textos]}``. En 1b el qid es el ``step_id``.
        description: explicación corta para los logs y las tablas.
        oracle: ``True`` si usa datos que no estarán en test (``gold_answers``, ``guidance``).
    """

    subtrack: str
    build: Callable[[str, str], dict[str, str | list[str]]]
    description: str
    oracle: bool = False


def html_to_text(text: str) -> str:
    """Quita HTML y normaliza espacios (``gold_answers`` viene en HTML).

    Args:
        text: texto HTML.

    Returns:
        Texto plano con los espacios colapsados.
    """
    return re.sub(r"\s+", " ", BeautifulSoup(text, "html.parser").get_text(" ")).strip()


def first_words(text: str, n: int) -> str:
    """Las ``n`` primeras palabras de un texto (separadas por espacios).

    Args:
        text: texto.
        n: número de palabras.

    Returns:
        Las palabras unidas con un espacio.
    """
    return " ".join(text.split()[:n])


# ----------------------------------------------------------------------------- 1a


def _per_query(fn: Callable[[dict, dict], str | list[str]]) -> Callable[[str, str], dict]:
    """Construye una variante 1a a partir de una función por consulta.

    Args:
        fn: ``fn(ejemplo, query_guidance) -> texto | [textos]``. ``query_guidance`` es
            ``{}`` si la consulta no tiene guidance; las variantes realistas lo ignoran.

    Returns:
        La función ``build(dominio, split)`` de la variante.
    """
    def build(domain: str, split: str) -> dict:
        guidance = load_guidance(domain, split)
        return {r["id"]: fn(r, guidance.get(r["id"], {}).get("query_guidance", {}))
                for r in load_examples(domain, split)}
    return build


def _title(r: dict) -> str:
    """Título de la consulta (primera línea, sin HTML)."""
    return split_query(r["query"])[0]


def _title_head(r: dict, n: int = 64) -> str:
    """Título + las ``n`` primeras palabras del cuerpo: una condensación sin LLM."""
    title, body = split_query(r["query"])
    return f"{title}. {first_words(body, n)}"


def _answer(r: dict) -> str:
    """Respuesta aceptada de StackExchange en texto plano (oráculo O1); ``""`` si no hay."""
    return html_to_text(r["gold_answers"][0]) if r["gold_answers"] else ""


def _normalized_tags(qg: dict) -> str:
    """Bloque estructurado al estilo *Normalized* del paper TEMPO (tabla 5).

    Args:
        qg: ``query_guidance`` de la consulta.

    Returns:
        ``"TEMPORAL_INTENT=...; CLASS=...; GRANULARITY=...; ANCHORS=[...]; EVENTS=[...]"``.
    """
    return (f"TEMPORAL_INTENT={qg.get('temporal_intent', '')}; "
            f"CLASS={qg.get('temporal_reasoning_class_primary', '')}; "
            f"GRANULARITY={qg.get('expected_granularity', '')}; "
            f"ANCHORS=[{', '.join(qg.get('key_time_anchors', []))}]; "
            f"EVENTS=[{', '.join(qg.get('query_temporal_events', []))}]")


def _anchors_events(qg: dict) -> str:
    """Anclas temporales y eventos de ``guidance`` como texto plano, separados por espacios."""
    return " ".join(qg.get("key_time_anchors", []) + qg.get("query_temporal_events", []))


# Variantes de 1a. Las que empiezan por O1/O2/O3 son oráculos de la Fase 2.
VARIANTS_1A: dict[str, Variant] = {
    "cruda": Variant("1a", _per_query(lambda r, qg: r["query"]), "consulta original con HTML (baseline oficial)"),
    "limpia": Variant("1a", _per_query(lambda r, qg: clean_query(r["query"])), "título + cuerpo sin HTML"),
    "titulo": Variant("1a", _per_query(lambda r, qg: _title(r)), "solo el título"),
    "titulo+inicio": Variant("1a", _per_query(lambda r, qg: _title_head(r)),
                             "título + 64 primeras palabras del cuerpo (condensación sin LLM)"),
    # Lista de dos textos: se busca con cada uno y se fusiona con RRF.
    "rrf(limpia,titulo)": Variant("1a", _per_query(lambda r, qg: [clean_query(r["query"]), _title(r)]),
                                  "RRF de la consulta limpia y del título"),
    # --- O1: techo de HyDE (respuesta aceptada de StackExchange como documento hipotético)
    "O1_respuesta": Variant("1a", _per_query(lambda r, qg: _answer(r)), "solo la respuesta aceptada", oracle=True),
    "O1_limpia+respuesta": Variant("1a", _per_query(lambda r, qg: f"{clean_query(r['query'])} {_answer(r)}"),
                                   "consulta limpia concatenada con la respuesta", oracle=True),
    "O1_rrf(limpia,respuesta)": Variant("1a", _per_query(lambda r, qg: [clean_query(r["query"]), _answer(r)]),
                                        "RRF de la consulta limpia y de la respuesta", oracle=True),
    # --- O2: tiempo estructurado y explícito (Normalized del paper)
    "O2_normalizada": Variant("1a", _per_query(lambda r, qg: f"{clean_query(r['query'])}\n{_normalized_tags(qg)}"),
                              "consulta limpia + etiquetas INTENT/CLASS/ANCHORS/EVENTS", oracle=True),
    "O2_limpia+anclas": Variant("1a", _per_query(lambda r, qg: f"{clean_query(r['query'])} {_anchors_events(qg)}"),
                                "consulta limpia + anclas y eventos en texto plano", oracle=True),
    # --- O3: reescritura hecha por un LLM (la de guidance)
    "O3_resumen": Variant("1a", _per_query(lambda r, qg: qg.get("query_summary", "")),
                          "query_summary de guidance", oracle=True),
    "O3_resumen+anclas": Variant("1a", _per_query(lambda r, qg: f"{qg.get('query_summary', '')} {_anchors_events(qg)}"),
                                 "query_summary + anclas + eventos", oracle=True),
    "O3_titulo+resumen": Variant("1a", _per_query(lambda r, qg: f"{_title(r)}. {qg.get('query_summary', '')}"),
                                 "título + query_summary", oracle=True),
    # Un texto por acción del plan de recuperación de guidance, más la consulta limpia.
    "O3_rrf(limpia,plan)": Variant("1a", _per_query(lambda r, qg: [clean_query(r["query"])]
                                                     + [f"{_title(r)}. {a['action']}" for a in qg.get("retrieval_plan", [])]),
                                   "RRF de la consulta limpia y de título + cada acción del retrieval_plan", oracle=True),
}


# ----------------------------------------------------------------------------- 1b

# Comienzos formulaicos de las instrucciones (EDA 5.1): 15 verbos inician el 97 % de los pasos.
# Incluye las formas compuestas ("search for", "identify and analyze", "cross-reference findings with").
_STEP_VERB_RE = re.compile(
    r"^(search(\s+for)?|identify(\s+and\s+analy[sz]e)?|cross-reference(\s+(findings|timelines|sources))?(\s+(with|of|to))?|"
    r"cross-check|locate|analy[sz]e|investigate|retrieve|verify|consult|examine|compare|review|filter(\s+results\s+to)?|"
    r"check|extract|compile|research|evaluate|trace)\b\s*",
    re.IGNORECASE,
)
# Ruido meta del generador: tipo de fuente y coletillas "focusing on temporal ...".
# Tres alternativas:
#   1. "(primary|official...) sources/records/documentation (on|of|about...)";
#   2. ", focusing on temporal ..." hasta la siguiente coma o punto;
#   3. frases hechas como "case studies or examples" o "explanations of".
_STEP_META_RE = re.compile(
    r"\b((primary|secondary|historical|official|authoritative|documented|reliable)\s+)*"
    r"(sources?|records|accounts|documentation|analyses|literature|resources)\b(\s+(on|of|about|for|that|explaining|discussing))?|"
    r",?\s*(focusing|emphasizing|with\s+a\s+focus)\s+on\s+temporal[^,.]*|"
    r"\b(case studies or examples|studies or papers|explanations( or [a-z ]+?)? of)\b",
    re.IGNORECASE,
)


def strip_step(instruction: str) -> str:
    """Quita el verbo imperativo inicial y las palabras meta de un paso (S2 sin LLM).

    Ejemplo: ``"Search for primary sources on the 1848 revolutions"`` →
    ``"the 1848 revolutions"``.

    Args:
        instruction: ``step_instruction`` del paso.

    Returns:
        La instrucción limpia, o la original si la limpieza la deja vacía.
    """
    text = _STEP_VERB_RE.sub("", instruction.strip())
    text = _STEP_META_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    # Conectores que quedan colgando al quitar la fuente ("primary sources or documenting ...").
    text = re.sub(r"^((or|and|specifically|,)\s+)+", "", text, flags=re.IGNORECASE)
    return text or instruction


def _per_step(fn: Callable[[dict, dict], str | list[str]]) -> Callable[[str, str], dict]:
    """Construye una variante 1b a partir de una función por paso.

    Args:
        fn: ``fn(consulta, paso) -> texto | [textos]``; ``consulta`` es la entrada de
            ``load_steps`` y ``paso`` uno de sus ``steps``.

    Returns:
        La función ``build(dominio, split)`` de la variante, con qids = ``step_id``.
    """
    def build(domain: str, split: str) -> dict:
        return {s["step_id"]: fn(r, s) for r in load_steps(domain, split) for s in r["steps"]}
    return build


# Variantes de 1b (ninguna es oráculo).
VARIANTS_1B: dict[str, Variant] = {
    "oficial": Variant("1b", _per_step(lambda r, s: f"{r['query']}\n\nStep: {s['step_instruction']}"),
                       "consulta cruda + 'Step:' + instrucción (baseline oficial)"),
    "paso": Variant("1b", _per_step(lambda r, s: s["step_instruction"]), "solo la instrucción"),
    "limpia+paso": Variant("1b", _per_step(lambda r, s: f"{clean_query(r['query'])} {s['step_instruction']}"),
                           "consulta limpia + instrucción"),
    "titulo+paso": Variant("1b", _per_step(lambda r, s: f"{split_query(r['query'])[0]}. {s['step_instruction']}"),
                           "título + instrucción"),
    "titulo+paso_limpio": Variant("1b", _per_step(lambda r, s: f"{split_query(r['query'])[0]}. {strip_step(s['step_instruction'])}"),
                                  "título + instrucción sin verbo ni palabras meta"),
    "rrf(paso,limpia)": Variant("1b", _per_step(lambda r, s: [s["step_instruction"], clean_query(r["query"])]),
                                "RRF de la instrucción y de la consulta limpia (S1)"),
    "rrf(titulo+paso,limpia)": Variant("1b", _per_step(lambda r, s: [f"{split_query(r['query'])[0]}. {s['step_instruction']}",
                                                                      clean_query(r["query"])]),
                                       "RRF de título + instrucción y de la consulta limpia"),
}

# Acceso por subtrack: VARIANTS["1a"]["limpia"].
VARIANTS = {"1a": VARIANTS_1A, "1b": VARIANTS_1B}
