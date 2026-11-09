"""Registro de variantes de consulta (qué texto se usa para buscar).

Cada variante es una función ``(dominio, split) -> {qid: texto | [textos]}``. Si una
consulta tiene varios textos, se busca con cada uno y los rankings se fusionan con
RRF (``scripts/run_experiment.py``).

Las variantes marcadas como oráculo usan ``gold_answers`` o ``guidance``, que no
estarán en test: solo sirven para medir techos (Fase 2 del plan).
"""

import re
from dataclasses import dataclass
from typing import Callable

from bs4 import BeautifulSoup

from reteco.data import clean_query, load_examples, load_guidance, load_steps, split_query


@dataclass(frozen=True)
class Variant:
    subtrack: str
    build: Callable[[str, str], dict[str, str | list[str]]]
    description: str
    oracle: bool = False


def html_to_text(text: str) -> str:
    """Quita HTML y normaliza espacios (``gold_answers`` viene en HTML)."""
    return re.sub(r"\s+", " ", BeautifulSoup(text, "html.parser").get_text(" ")).strip()


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


# ----------------------------------------------------------------------------- 1a


def _per_query(fn: Callable[[dict, dict], str | list[str]]) -> Callable[[str, str], dict]:
    """Construye una variante 1a a partir de ``fn(ejemplo, query_guidance)``."""
    def build(domain: str, split: str) -> dict:
        guidance = load_guidance(domain, split)
        return {r["id"]: fn(r, guidance.get(r["id"], {}).get("query_guidance", {}))
                for r in load_examples(domain, split)}
    return build


def _title(r: dict) -> str:
    return split_query(r["query"])[0]


def _title_head(r: dict, n: int = 64) -> str:
    title, body = split_query(r["query"])
    return f"{title}. {first_words(body, n)}"


def _answer(r: dict) -> str:
    return html_to_text(r["gold_answers"][0]) if r["gold_answers"] else ""


def _normalized_tags(qg: dict) -> str:
    """Bloque estructurado al estilo *Normalized* del paper TEMPO (tabla 5)."""
    return (f"TEMPORAL_INTENT={qg.get('temporal_intent', '')}; "
            f"CLASS={qg.get('temporal_reasoning_class_primary', '')}; "
            f"GRANULARITY={qg.get('expected_granularity', '')}; "
            f"ANCHORS=[{', '.join(qg.get('key_time_anchors', []))}]; "
            f"EVENTS=[{', '.join(qg.get('query_temporal_events', []))}]")


def _anchors_events(qg: dict) -> str:
    return " ".join(qg.get("key_time_anchors", []) + qg.get("query_temporal_events", []))


VARIANTS_1A: dict[str, Variant] = {
    "cruda": Variant("1a", _per_query(lambda r, qg: r["query"]), "consulta original con HTML (baseline oficial)"),
    "limpia": Variant("1a", _per_query(lambda r, qg: clean_query(r["query"])), "título + cuerpo sin HTML"),
    "titulo": Variant("1a", _per_query(lambda r, qg: _title(r)), "solo el título"),
    "titulo+inicio": Variant("1a", _per_query(lambda r, qg: _title_head(r)),
                             "título + 64 primeras palabras del cuerpo (condensación sin LLM)"),
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
    "O3_rrf(limpia,plan)": Variant("1a", _per_query(lambda r, qg: [clean_query(r["query"])]
                                                     + [f"{_title(r)}. {a['action']}" for a in qg.get("retrieval_plan", [])]),
                                   "RRF de la consulta limpia y de título + cada acción del retrieval_plan", oracle=True),
}


# ----------------------------------------------------------------------------- 1b

# Comienzos formulaicos de las instrucciones (EDA 5.1): 15 verbos inician el 97 % de los pasos.
_STEP_VERB_RE = re.compile(
    r"^(search(\s+for)?|identify(\s+and\s+analy[sz]e)?|cross-reference(\s+(findings|timelines|sources))?(\s+(with|of|to))?|"
    r"cross-check|locate|analy[sz]e|investigate|retrieve|verify|consult|examine|compare|review|filter(\s+results\s+to)?|"
    r"check|extract|compile|research|evaluate|trace)\b\s*",
    re.IGNORECASE,
)
# Ruido meta del generador: tipo de fuente y coletillas "focusing on temporal ...".
_STEP_META_RE = re.compile(
    r"\b((primary|secondary|historical|official|authoritative|documented|reliable)\s+)*"
    r"(sources?|records|accounts|documentation|analyses|literature|resources)\b(\s+(on|of|about|for|that|explaining|discussing))?|"
    r",?\s*(focusing|emphasizing|with\s+a\s+focus)\s+on\s+temporal[^,.]*|"
    r"\b(case studies or examples|studies or papers|explanations( or [a-z ]+?)? of)\b",
    re.IGNORECASE,
)


def strip_step(instruction: str) -> str:
    """Quita el verbo imperativo inicial y las palabras meta de un paso (S2 sin LLM)."""
    text = _STEP_VERB_RE.sub("", instruction.strip())
    text = _STEP_META_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    # Conectores que quedan colgando al quitar la fuente ("primary sources or documenting ...").
    text = re.sub(r"^((or|and|specifically|,)\s+)+", "", text, flags=re.IGNORECASE)
    return text or instruction


def _per_step(fn: Callable[[dict, dict], str | list[str]]) -> Callable[[str, str], dict]:
    """Construye una variante 1b a partir de ``fn(consulta, paso)``."""
    def build(domain: str, split: str) -> dict:
        return {s["step_id"]: fn(r, s) for r in load_steps(domain, split) for s in r["steps"]}
    return build


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

VARIANTS = {"1a": VARIANTS_1A, "1b": VARIANTS_1B}
