"""Grafo temporal de dos niveles de TG-RAG (``papers/2510.13590v1.pdf`` §III-A).

- **Extracción de hechos** ``(entidad₁, entidad₂, relación, τ)`` por documento:
  - ``rules``: sin LLM. Oraciones, entidades capitalizadas y expresiones temporales
    con ``reteco.temporal``. Una arista por par de entidades consecutivas de la oración,
    con la oración como descripción.
  - ``llm``: prompt propio (el paper no publica el de extracción) que devuelve hechos
    JSON con el tiempo normalizado.
- **Grafo:** las entidades son los nodos; las aristas llevan τ, y las aristas paralelas
  del mismo par en fechas distintas se conservan. ``time_index`` enlaza cada nodo de la
  jerarquía de tiempo con sus aristas (el nivel superior del paper).

Los hechos se guardan por documento en ``outputs/exp/tgrag/facts/<extractor>/<dominio>.jsonl``.
Como la extracción es incremental, si el pool crece solo se procesan los documentos nuevos.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from reteco.data import EXP_DIR
from reteco.temporal import URL_RE, TimeSpan, ancestors, extract_spans, parse_normalized, span_nodes

TG_DIR = EXP_DIR / "tgrag"
CHUNK_WORDS = 800       # ~1,150 tokens: el chunk de 1,200 tokens del paper
MAX_CHUNKS = 4          # como en reteco.dense: se cubre el inicio de los documentos largos
MAX_SENTENCES = 80      # por chunk (extractor de reglas)
MAX_ENTITIES = 6        # por oración
MAX_ENTITY_TOKENS = 5   # sin contar conectores; más largas suelen ser títulos o referencias fusionadas
MAX_TIMES = 3           # por oración
DESC_CHARS = 400


@dataclass
class Fact:
    chunk: int
    e1: str
    e2: str
    relation: str
    time: TimeSpan | None
    description: str

    def to_row(self) -> list:
        return [self.chunk, self.e1, self.e2, self.relation, self.time.to_str() if self.time else None, self.description]

    @staticmethod
    def from_row(row: list) -> "Fact":
        chunk, e1, e2, rel, t, desc = row
        return Fact(chunk, e1, e2, rel, TimeSpan.from_str(t) if t else None, desc)


def chunk_text(text: str) -> list[str]:
    """Trozos de ``CHUNK_WORDS`` palabras (como mucho ``MAX_CHUNKS``)."""
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[s:s + CHUNK_WORDS]) for s in range(0, len(words), CHUNK_WORDS)][:MAX_CHUNKS]


# ----------------------------------------------------------------------------- extractor de reglas

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"“'(\[]?[A-Z0-9])|\s*\n+\s*")
# Acrónimos con puntos ("U.S.", "J.D.") como un solo token; si no, palabras.
_TOKEN_RE = re.compile(r"(?:[A-Z]\.){2,}|[A-Za-zÀ-ÖØ-öø-ÿ][\w'’&-]*")
# Conectores que pueden ir dentro de un nombre ("Bank of England", "Leonardo da Vinci").
# Sin "and", "&" ni "y": unían nombres distintos ("Tyler and Cameron Winklevoss").
_CONNECTORS = {"of", "the", "de", "la", "le", "del", "von", "van", "der", "den", "du", "di", "da", "al", "bin", "ibn"}
# Tratamientos: "Mr. Smith" → SMITH (el punto corta la entidad y el tratamiento se descarta).
_TITLES = {"mr", "mrs", "ms", "dr", "prof", "sir", "st", "jr", "sr", "rev", "gen", "col", "capt", "lt", "hon"}
# Palabras de discurso que abren oraciones y no son nombres ("Finally, …", "Another …").
_DISCOURSE = {
    "another", "finally", "moreover", "furthermore", "further", "firstly", "secondly", "thirdly", "lastly",
    "consequently", "hence", "therefore", "accordingly", "additionally", "similarly", "likewise", "otherwise",
    "besides", "meanwhile", "instead", "indeed", "nevertheless", "nonetheless", "overall", "specifically",
    "generally", "typically", "notably", "importantly", "interestingly", "unfortunately", "fortunately",
    "recently", "currently", "previously", "initially", "eventually", "ultimately", "subsequently", "still",
    "yet", "even", "just", "only", "perhaps", "maybe", "please", "thank", "thanks", "according", "despite",
    "unlike", "like", "using", "given", "based", "following", "regarding", "including", "whether", "every",
    "any", "few", "several", "more", "less", "much", "very", "let", "once", "prior", "two", "three", "four",
    "five", "six", "seven", "eight", "nine", "ten", "current", "fixed", "added", "removed", "updated",
    "understanding", "consider", "imagine", "suppose", "note", "example", "e.g", "i.e", "yes", "ok", "okay",
    "hi", "hello", "dear", "welcome", "great", "good", "nice", "sure", "well", "oh", "wow",
}
_STOP_CAPS = _DISCOURSE | _TITLES | {
    "i", "a", "an", "the", "this", "that", "these", "those", "it", "its", "he", "she", "they", "we", "you", "his", "her",
    "their", "our", "my", "your", "in", "on", "at", "for", "but", "and", "or", "if", "when", "where", "while", "as", "by",
    "with", "from", "to", "of", "after", "before", "during", "since", "until", "there", "here", "what", "which", "who",
    "why", "how", "however", "also", "then", "thus", "so", "some", "many", "most", "all", "both", "each", "one", "no",
    "not", "yes", "is", "was", "are", "were", "be", "been", "has", "have", "had", "do", "does", "did", "can", "could",
    "will", "would", "may", "might", "should", "must", "although", "though", "because", "other", "such", "first",
    "later", "today", "now", "see", "note", "notes", "page", "chapter", "figure", "table", "edit", "retrieved",
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "century", "ad", "bc", "bce", "ce",
}


def normalize_entity(name: str) -> str:
    """Nombre canónico de una entidad: mayúsculas, sin posesivo, sin puntos de acrónimo ni puntuación de borde."""
    name = re.sub(r"['’]s\b", "", name.strip())
    name = re.sub(r"\b([A-Z])\.(?=[A-Z]\.|\s|$)", r"\1", name)       # "U.S." → "US", "J.D. Dean" → "JD Dean"
    name = re.sub(r"\s+", " ", name).strip(" .,;:-'\"“”()[]")
    return name.upper()


# ----------------------------------------------------------------------------- filtros de oración
# Ver REVISION_EXTRACCION_TGRAG.md §3.4.

_BBCODE_RE = re.compile(r"\[/?(?:url|b|i|img|link|quote|code)\b", re.IGNORECASE)
# Sin ";" final: las enumeraciones legales terminan así ("…shall report;").
_CODE_RE = re.compile(r"//|/\*|\*/|#include|\bdef\s+\w+\(|\breturn\b.*;|[{}]\s*$|=>|->|==|!=|\+=|&&|\|\|")
_CODE_CHARS = set("{}[]<>=;|\\/*+#$@~^`")


def is_prose(sentence: str) -> bool:
    """¿Es texto en prosa? Descarta spam con BBCode, código y líneas casi todo en mayúsculas.

    - BBCode (``[url=…]``, ``[b]``): spam de foros; las entidades y fechas son basura.
    - Código: marcas típicas (``//``, ``==``, ``=>``, ``;`` final…) o >10 % de símbolos.
    - Mayúsculas: con ≥ 20 letras, más del 60 % en mayúscula (titulares, cabeceras de tabla,
      volcados de ensamblador). Ahí la heurística de "palabra capitalizada = nombre" no vale.
    """
    if _BBCODE_RE.search(sentence):
        return False
    text = URL_RE.sub(" ", sentence)              # "http://" no debe contar como comentario "//"
    if _CODE_RE.search(text):
        return False
    stripped = text.replace(" ", "")
    if stripped and sum(c in _CODE_CHARS for c in stripped) / len(stripped) > 0.10:
        return False
    letters = [c for c in text if c.isalpha()]
    return not (len(letters) >= 20 and sum(c.isupper() for c in letters) / len(letters) > 0.60)


def mid_sentence_caps(sentences: list[str]) -> set[str]:
    """Palabras que aparecen capitalizadas en mitad de alguna oración (en minúsculas).

    Sirve para decidir si la primera palabra de una oración es un nombre: "Monero is…"
    se acepta si "Monero" aparece capitalizada en otra posición del mismo chunk; "Another
    crisis…" no.
    """
    seen = set()
    for sent in sentences:
        for m in list(_TOKEN_RE.finditer(sent))[1:]:
            tok = m.group()
            if tok[0].isupper():
                seen.add(tok.lower())
    return seen


def _gap_breaks(gap: str, prev_tok: str) -> bool:
    """¿Corta la entidad el texto entre dos tokens? Cualquier puntuación corta, salvo el
    punto de una inicial ("John F. Kennedy")."""
    if gap.strip() == "":
        return False
    return not (gap.strip() == "." and len(prev_tok) == 1 and prev_tok.isupper())


def rule_entities(sentence: str, known: set[str] | None = None) -> list[str]:
    """Secuencias capitalizadas (con conectores internos como "of", "de") y acrónimos.

    Args:
        sentence: oración (las URLs se ignoran).
        known: palabras vistas capitalizadas en mitad de oración en el mismo chunk
            (``mid_sentence_caps``). Si se da, una entidad de **una sola palabra** al
            inicio de la oración solo se acepta si está en ``known`` o es un acrónimo.
            Con ``None`` se mantiene el comportamiento anterior (solo listas de exclusión).

    Reglas (REVISION_EXTRACCION_TGRAG.md §3.4):
        - la puntuación entre tokens corta la entidad (comas, punto y coma, paréntesis…);
        - "and" / "&" no unen nombres;
        - se quitan del inicio palabras de discurso, tratamientos y palabras vacías;
        - se descartan entidades de más de ``MAX_ENTITY_TOKENS`` palabras (sin conectores).
    """
    sentence = URL_RE.sub(lambda m: " " * len(m.group()), sentence)
    tokens = [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(sentence)]
    ents, cur = [], []
    for i, (tok, s, _) in enumerate(tokens):
        if cur and _gap_breaks(sentence[tokens[i - 1][2]:s], tokens[i - 1][0]):
            ents.append(cur)
            cur = []
        if tok[0].isupper():
            cur.append((tok, i))
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if (cur and tok.lower() in _CONNECTORS and nxt is not None and nxt[0][:1].isupper()
                and not _gap_breaks(sentence[tokens[i][2]:nxt[1]], tok)):
            cur.append((tok, i))
            continue
        if cur:
            ents.append(cur)
        cur = []
    if cur:
        ents.append(cur)
    out = []
    for toks in ents:
        while toks and toks[0][0].lower() in _STOP_CAPS:
            toks = toks[1:]
        while toks and toks[-1][0].lower() in _CONNECTORS:
            toks = toks[:-1]
        if not toks:
            continue
        words = [t for t, _ in toks]
        if sum(w.lower() not in _CONNECTORS for w in words) > MAX_ENTITY_TOKENS:
            continue
        if (known is not None and len(words) == 1 and toks[0][1] == 0
                and not words[0].isupper() and words[0].lower() not in known):
            continue
        name = normalize_entity(" ".join(words))
        if len(name) < 2 or (len(words) == 1 and words[0].lower() in _STOP_CAPS):
            continue
        out.append(name)
    return list(dict.fromkeys(out))[:MAX_ENTITIES]


def extract_rules(text: str, inherit_time: bool = False, drop_time: tuple[str, ...] = ()) -> list[Fact]:
    """Hechos de un documento con reglas.

    Args:
        text: contenido del documento.
        inherit_time: si una oración no tiene tiempo, hereda el último τ visto en el
            mismo chunk ("In 1834 he moved to X. There he founded Y" → Y hereda 1834).
        drop_time: papeles de fecha que no se usan como τ (``reteco.temporal.ROLES``:
            ``"citation"``, ``"metadata"``). Por defecto se usan todas las fechas.

    Returns:
        Hechos con relación ``"co-occurs"``. Las oraciones sin tiempo dan hechos con
        ``time=None``, como los tripletes no temporales de la variante genérica del paper.
        Las oraciones que no son prosa (``is_prose``) no dan hechos.
    """
    facts = []
    for c, chunk in enumerate(chunk_text(text)):
        last_times: list[TimeSpan] = []
        sents = [s for s in _SENT_RE.split(chunk) if s and s.strip() and is_prose(s)][:MAX_SENTENCES]
        known = mid_sentence_caps(sents)
        for sent in sents:
            times = [span for _, _, span in extract_spans(sent, drop=drop_time)][:MAX_TIMES]
            if times:
                last_times = times
            elif inherit_time:
                times = last_times
            ents = rule_entities(sent, known)
            if not ents:
                continue
            pairs = list(zip(ents, ents[1:])) or [(ents[0], ents[0])]
            desc = sent.strip()[:DESC_CHARS]
            for t in times or [None]:
                facts.extend(Fact(c, a, b, "co-occurs", t, desc) for a, b in pairs)
    return facts


# ----------------------------------------------------------------------------- extractor LLM

EXTRACT_PROMPT_ID = "tg_extract_v1"
EXTRACT_PROMPT = """-Goal-
Given a text passage, extract temporal facts: relations between two entities, each anchored to the time when the relation holds or the event happened.

-Steps-
1. Identify the time expressions in the passage: explicit dates, years, decades, centuries, ranges, and times that are clearly implied by the context.
2. Identify the non-temporal entities: people, organizations, places, events, objects, artifacts, technologies, laws, concepts.
3. For each relation between two entities, output one fact. If the same relation holds at different times, output one fact per time.
4. Also output the most important relations that have no time, with "time": null.

-Time format-
Use one of: YYYY (year; negative for BCE, e.g. -300), YYYY-MM, YYYY-MM-DD, YYYY-Qn, YYY0s (decade, e.g. 1830s), YY00s (century, e.g. 1800s), CNN (century number, e.g. C19; C-5 is the 5th century BCE), YYYY/YYYY (range), present.

-Output-
A JSON list with at most {max_facts} items. Each item:
{"subject": "<entity>", "object": "<entity>", "relation": "<short verb phrase>", "time": "<normalized time or null>", "description": "<one self-contained sentence stating the fact, including its time>"}
Output only the JSON list.

-Example-
Text: The Treaty of Paris, signed in September 1783 by Britain and the United States, ended the American Revolutionary War. Benjamin Franklin led the American delegation.
Output: [{"subject": "Treaty of Paris", "object": "American Revolutionary War", "relation": "ended", "time": "1783-09", "description": "The Treaty of Paris ended the American Revolutionary War in September 1783."}, {"subject": "Britain", "object": "United States", "relation": "signed treaty with", "time": "1783-09", "description": "Britain and the United States signed the Treaty of Paris in September 1783."}, {"subject": "Benjamin Franklin", "object": "American delegation", "relation": "led", "time": "1783", "description": "Benjamin Franklin led the American delegation to the Treaty of Paris in 1783."}]

-Real Data-
Text: {text}
Output:"""


def extract_prompt(chunk: str, max_facts: int = 15) -> str:
    return EXTRACT_PROMPT.replace("{max_facts}", str(max_facts)).replace("{text}", chunk)


def parse_llm_facts(output: str, chunk: int) -> list[Fact]:
    """Hechos de la respuesta del LLM; ignora los elementos mal formados."""
    from reteco.llm import parse_json_block
    data = parse_json_block(output)
    if isinstance(data, dict):
        data = [data]
    facts = []
    for item in data or []:
        if not isinstance(item, dict):
            continue
        e1, e2 = normalize_entity(str(item.get("subject") or "")), normalize_entity(str(item.get("object") or ""))
        if not e1 or not e2:
            continue
        rel = str(item.get("relation") or "related to").strip()
        desc = str(item.get("description") or f"{e1} {rel} {e2}").strip()[:DESC_CHARS]
        facts.append(Fact(chunk, e1, e2, rel, parse_normalized(item.get("time")), desc))
    return facts


def extract_llm(text: str, llm, doc_id: str, max_facts: int = 15) -> list[Fact]:
    """Hechos de un documento con el LLM: una llamada por chunk."""
    facts = []
    for c, chunk in enumerate(chunk_text(text)):
        if chunk.strip():
            out = llm.generate(EXTRACT_PROMPT_ID, f"{doc_id}#{c}", extract_prompt(chunk, max_facts))
            facts.extend(parse_llm_facts(out, c))
    return facts


# ----------------------------------------------------------------------------- E/S


def facts_path(extractor: str, domain: str) -> Path:
    return TG_DIR / "facts" / extractor / f"{domain}.jsonl"


def done_doc_ids(extractor: str, domain: str) -> set[str]:
    path = facts_path(extractor, domain)
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["doc_id"] for line in f if line.strip()}


def append_facts(extractor: str, domain: str, doc_id: str, facts: list[Fact], f=None) -> None:
    """Añade los hechos de un documento (también si no tiene ninguno, para no repetirlo)."""
    line = json.dumps({"doc_id": doc_id, "facts": [x.to_row() for x in facts]}, ensure_ascii=False) + "\n"
    if f is not None:
        f.write(line)
        return
    path = facts_path(extractor, domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def load_facts(extractor: str, domain: str, doc_ids: set[str] | None = None) -> dict[str, list[Fact]]:
    """Hechos por documento (solo los de ``doc_ids`` si se indica)."""
    out = {}
    with open(facts_path(extractor, domain), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if doc_ids is None or r["doc_id"] in doc_ids:
                out[r["doc_id"]] = [Fact.from_row(x) for x in r["facts"]]
    return out


def pool_path(subtrack: str, pool: str, domain: str, split: str) -> Path:
    return TG_DIR / "pools" / subtrack / pool / f"{domain}__{split}.json"


def load_pool(subtrack: str, pool: str, domain: str, split: str) -> dict[str, list[str]]:
    """Candidatos por consulta ``{qid: [doc_id, ...]}`` (en orden de la primera etapa)."""
    with open(pool_path(subtrack, pool, domain, split), encoding="utf-8") as f:
        return json.load(f)


def qtime_path(method: str, subtrack: str, domain: str, split: str) -> Path:
    return TG_DIR / "qtime" / method / subtrack / f"{domain}__{split}.json"


def load_qtime(method: str, subtrack: str, domain: str, split: str):
    """Alcance temporal ``T_q`` por consulta: ``{qid: [Scope, ...]}``."""
    from reteco.temporal import Scope
    with open(qtime_path(method, subtrack, domain, split), encoding="utf-8") as f:
        return {qid: [Scope.from_str(s) for s in scopes] for qid, scopes in json.load(f).items()}


# ----------------------------------------------------------------------------- grafo bi-nivel


class TemporalGraph:
    """Grafo temporal de un dominio (restringido a los documentos del pool).

    Atributos (arrays alineados por arista, agrupadas por documento):
        ``e1``, ``e2``: índices de entidad; ``doc``: índice de documento; ``chunk``;
        ``times``: τ de cada arista (``TimeSpan`` o ``None``); ``desc``: índice en
        ``descriptions`` (descripciones únicas, para calcular cada embedding una vez).
        ``doc_ptr``: las aristas del documento ``i`` son ``doc_ptr[i]:doc_ptr[i+1]``.
        ``time_index``: nodo de tiempo → aristas (incluidos los ancestros), perezoso.
    """

    def __init__(self, facts_by_doc: dict[str, list[Fact]]):
        self.doc_ids = list(facts_by_doc)
        self.doc_index = {d: i for i, d in enumerate(self.doc_ids)}
        self.entities: list[str] = []
        ent_index: dict[str, int] = {}
        self.descriptions: list[str] = []
        desc_index: dict[str, int] = {}
        e1, e2, doc, chunk, desc, self.times, ptr = [], [], [], [], [], [], [0]
        for i, d in enumerate(self.doc_ids):
            for fact in facts_by_doc[d]:
                e1.append(ent_index.setdefault(fact.e1, len(ent_index)))
                e2.append(ent_index.setdefault(fact.e2, len(ent_index)))
                desc.append(desc_index.setdefault(fact.description, len(desc_index)))
                doc.append(i)
                chunk.append(fact.chunk)
                self.times.append(fact.time)
            ptr.append(len(e1))
        self.entities = list(ent_index)
        self.descriptions = list(desc_index)
        self.e1, self.e2 = np.array(e1, np.int32), np.array(e2, np.int32)
        self.doc, self.chunk = np.array(doc, np.int32), np.array(chunk, np.int16)
        self.desc = np.array(desc, np.int32)
        self.doc_ptr = np.array(ptr, np.int64)
        self._adj = None
        self._time_index = None

    @property
    def time_index(self) -> dict[str, list[int]]:
        """Enlaces entre niveles: nodo de tiempo (y sus ancestros) → aristas. Se construye al pedirlo."""
        if self._time_index is None:
            index: dict[str, list[int]] = defaultdict(list)
            for k, t in enumerate(self.times):
                if t is None:
                    continue
                for node in span_nodes(t):
                    for key in (node, *ancestors(node)):
                        index[key].append(k)
            self._time_index = dict(index)
        return self._time_index

    @property
    def n_edges(self) -> int:
        return len(self.e1)

    def edges_of(self, doc_idx: list[int] | np.ndarray) -> np.ndarray:
        """Índices de las aristas extraídas de los documentos indicados."""
        parts = [np.arange(self.doc_ptr[i], self.doc_ptr[i + 1]) for i in doc_idx]
        return np.concatenate(parts) if parts else np.array([], np.int64)

    def adjacency(self) -> sp.csr_matrix:
        """Matriz de adyacencia simétrica entre entidades (aristas paralelas sumadas)."""
        if self._adj is None:
            n = len(self.entities)
            a = sp.coo_matrix((np.ones(self.n_edges, np.float32), (self.e1, self.e2)), shape=(n, n)).tocsr()
            self._adj = (a + a.T).tocsr()
        return self._adj

    def stats(self) -> dict:
        timed = sum(t is not None for t in self.times)
        return {"docs": len(self.doc_ids), "entidades": len(self.entities), "aristas": self.n_edges,
                "aristas_con_tiempo": timed, "descripciones": len(self.descriptions)}
