"""Rutas del proyecto y cargadores de datos del Track 1 (TEMPO).

Copia los cargadores del notebook ``baseline_track1_tempo.ipynb`` para no repetirlos
en cada script, y añade la normalización de las etiquetas temporales de ``guidance``.

Estructura de los datos (``data/RETECO-SemEval2027/track1_tempo/<dominio>/``):

- ``documents.jsonl``: corpus del dominio, un ``{"id", "content"}`` por línea.
- ``examples_<split>.jsonl``: consultas de 1a con sus gold.
- ``steps_<split>.jsonl``: consultas de 1b, divididas en pasos con sus propios gold.
- ``guidance_<split>.jsonl``: anotaciones temporales de cada consulta. Solo existen
  para train/dev y **nunca se usan en inferencia**: solo en análisis y oráculos.
- ``qrels_<split>.txt`` / ``qrels_steps_<split>.txt``: juicios en formato TREC.

Las funciones que leen archivos usan ``functools.cache``: cada archivo se lee una vez
por proceso y las llamadas siguientes devuelven el mismo objeto (no modificarlo).
"""

import html
import json
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

from bs4 import BeautifulSoup


def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    """Sube por los directorios hasta encontrar el ``pyproject.toml`` del proyecto.

    Así las rutas funcionan igual en el Mac y en el servidor, y desde cualquier
    directorio de trabajo (scripts, notebooks, pruebas).

    Args:
        start: ruta desde la que empezar a subir (por defecto, este archivo).

    Returns:
        El primer directorio que contiene ``pyproject.toml``.

    Raises:
        FileNotFoundError: si no hay ningún ``pyproject.toml`` por encima de ``start``.
    """
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("No se encontró pyproject.toml en ningún directorio padre")


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "RETECO-SemEval2027"
T1_DIR = DATA_DIR / "track1_tempo"     # un subdirectorio por dominio
EDA_DIR = ROOT / "outputs" / "eda"      # caché del EDA (índices BM25 ya construidos)
EXP_DIR = ROOT / "outputs" / "exp"      # runs y resultados de los experimentos

# Semilla global de los muestreos (train-mini, revisiones), fija para que todo sea reproducible.
SEED = 20270101
# Los 13 dominios del Track 1, en orden alfabético: cada subdirectorio de T1_DIR es uno.
DOMAINS = sorted(p.name for p in T1_DIR.iterdir() if p.is_dir())


# ----------------------------------------------------------------------------- lectura


def read_jsonl(path: Path) -> list[dict]:
    """Lee un archivo JSON Lines completo.

    Args:
        path: ruta del archivo; cada línea no vacía es un objeto JSON.

    Returns:
        Lista con un diccionario por línea, en el orden del archivo.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_documents(domain: str):
    """Recorre el corpus de un dominio sin cargarlo entero en memoria.

    ``history`` tiene ~200 k documentos, así que se lee línea a línea.

    Args:
        domain: nombre del dominio (uno de ``DOMAINS``).

    Yields:
        Un diccionario ``{"id": str, "content": str}`` por documento, en el orden del archivo.
        Ese orden es el que usa el baseline oficial para desempatar scores.
    """
    with open(T1_DIR / domain / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@cache
def load_examples(domain: str, split: str) -> list[dict]:
    """Consultas del subtrack 1a.

    Args:
        domain: dominio.
        split: ``"train"``, ``"dev"`` o ``"test"``.

    Returns:
        Lista de ``{id, query, gold_ids, gold_answers}``. ``query`` es el texto crudo
        (título + cuerpo HTML). ``gold_answers`` solo se usa en oráculos.
    """
    return read_jsonl(T1_DIR / domain / f"examples_{split}.jsonl")


@cache
def load_steps(domain: str, split: str) -> list[dict]:
    """Consultas del subtrack 1b, divididas en pasos.

    Args:
        domain: dominio.
        split: ``"train"``, ``"dev"`` o ``"test"``.

    Returns:
        Lista de ``{id, query, steps: [{step_id, step, step_instruction, gold_ids}]}``.
        En 1b se evalúa cada paso por separado: el qid de evaluación es ``step_id``.
    """
    return read_jsonl(T1_DIR / domain / f"steps_{split}.jsonl")


@cache
def load_guidance(domain: str, split: str) -> dict[str, dict]:
    """Anotaciones temporales de cada consulta (solo para análisis y oráculos).

    Args:
        domain: dominio.
        split: ``"train"`` o ``"dev"`` (no existen para test).

    Returns:
        ``{id_consulta: {"query_guidance": {...}, "gold_passage_annotations": [...]}}``.
    """
    return {g["id"]: g for g in read_jsonl(T1_DIR / domain / f"guidance_{split}.jsonl")}


@cache
def load_qrels(domain: str, split: str, subtrack: str = "1a") -> dict[str, dict[str, int]]:
    """Juicios de relevancia en formato TREC.

    Cada línea del archivo es ``qid 0 docid relevancia``.

    Args:
        domain: dominio.
        split: ``"train"`` o ``"dev"``.
        subtrack: ``"1a"`` (por consulta) o ``"1b"`` (por paso).

    Returns:
        ``{qid: {doc_id: relevancia}}``; en 1b el qid es el ``step_id``.
    """
    name = f"qrels_{split}.txt" if subtrack == "1a" else f"qrels_steps_{split}.txt"
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(T1_DIR / domain / name, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, _, docid, rel = line.split()   # la segunda columna (iteración TREC) no se usa
                qrels[qid][docid] = int(rel)
    return dict(qrels)


def parent_query_id(qid: str) -> str:
    """Id de la consulta madre de un paso de 1b.

    Args:
        qid: ``step_id`` (``"124973_5_step2"``) o id de consulta (se devuelve igual).

    Returns:
        El id sin el sufijo ``_stepN`` (``"124973_5"``).
    """
    return re.sub(r"_step\d+$", "", qid)


# ----------------------------------------------------------------------------- limpieza


def split_query(raw: str) -> tuple[str, str]:
    """Separa una consulta StackExchange en título y cuerpo, sin HTML.

    Las consultas llegan como ``"<título>\\n<p>cuerpo html</p>..."``: la primera línea
    es el título en texto plano y el resto es el cuerpo en HTML.

    Args:
        raw: texto crudo de la consulta.

    Returns:
        ``(título, cuerpo)``, los dos con entidades HTML decodificadas y espacios
        colapsados. El cuerpo es ``""`` si la consulta solo tiene título.
    """
    title, _, body_html = raw.partition("\n")
    # get_text(" ") pone un espacio entre etiquetas para que no se peguen palabras de párrafos distintos.
    body = BeautifulSoup(body_html, "html.parser").get_text(" ")
    norm = lambda s: re.sub(r"\s+", " ", html.unescape(s)).strip()
    return norm(title), norm(body)


@cache
def clean_query(raw: str) -> str:
    """Consulta como texto plano (variante ``limpia``).

    Args:
        raw: texto crudo de la consulta.

    Returns:
        ``"título. cuerpo"``, o solo el título si no hay cuerpo.
    """
    title, body = split_query(raw)
    return f"{title}. {body}" if body else title


# ----------------------------------------------------------------------------- etiquetas temporales

# Clases primarias de razonamiento temporal con al menos ~60 consultas en train; las
# demás (raras o mal escritas en guidance) se agrupan en "other".
MAIN_CLASSES = (
    "event_analysis_and_localization",
    "time_period_contextualization",
    "origins_evolution_comparative_analysis",
    "trends_changes_and_cross_period",
    "event_verification_and_authenticity",
    "materials_artifacts_and_provenance",
)
# Abreviaturas usadas en las tablas de resultados (EAL, TPC...).
CLASS_ABBR = {
    "event_analysis_and_localization": "EAL",
    "time_period_contextualization": "TPC",
    "origins_evolution_comparative_analysis": "OEC",
    "trends_changes_and_cross_period": "TCP",
    "event_verification_and_authenticity": "EVA",
    "materials_artifacts_and_provenance": "MAP",
    "other": "OTH",
}
# temporal_intent trae a veces nombres de clase; solo se conservan las intenciones reales.
# Las variantes de orden relativo (before_after, after, since) se agrupan en "order".
INTENT_MAP = {
    "when": "when", "period_definition": "period_definition", "timeline": "timeline",
    "ongoing_status": "ongoing_status", "duration": "duration",
    "order": "order", "before_after": "order", "after": "order", "since": "order",
}


def normalize_class(raw: str) -> str:
    """Normaliza la clase primaria de ``guidance``.

    Args:
        raw: valor de ``temporal_reasoning_class_primary``.

    Returns:
        El mismo valor si está en ``MAIN_CLASSES``; si no, ``"other"``.
    """
    return raw if raw in MAIN_CLASSES else "other"


def normalize_intent(raw: str) -> str:
    """Normaliza la intención temporal de ``guidance``.

    Args:
        raw: valor de ``temporal_intent``.

    Returns:
        ``when``, ``period_definition``, ``timeline``, ``ongoing_status``, ``duration``,
        ``order`` u ``"other"`` (ver ``INTENT_MAP``).
    """
    return INTENT_MAP.get(raw, "other")


def train_mini_ids(domain: str, split: str = "train", n: int = 40) -> set[str]:
    """Subconjunto ``train-mini`` (PLAN_EXPERIMENTOS.md §1): hasta ``n`` consultas por dominio.

    Sirve para probar cosas caras (LLM) en pocas consultas sin sesgar la mezcla de clases.
    Muestreo estratificado por clase primaria con reparto proporcional (método de los
    restos mayores) y semilla fija por dominio. La clase solo se usa para muestrear,
    nunca en inferencia.

    Args:
        domain: dominio.
        split: split del que se muestrea.
        n: tamaño máximo del subconjunto.

    Returns:
        Conjunto de ids de consulta. Si el dominio tiene ``n`` consultas o menos, todas.
    """
    import random

    labels = query_labels(domain, split)
    qids = [r["id"] for r in load_examples(domain, split)]
    if len(qids) <= n:
        return set(qids)
    # Agrupa las consultas por clase primaria.
    groups: dict[str, list[str]] = defaultdict(list)
    for q in qids:
        groups[labels.get(q, {}).get("clase", "other")].append(q)
    # La semilla depende del dominio y el split: muestras distintas pero reproducibles.
    rng = random.Random(f"{SEED}-{domain}-{split}")
    # Restos mayores: cada clase recibe la parte entera de su cuota proporcional y las
    # plazas que faltan hasta n van a las clases con mayor parte decimal.
    quota = {g: n * len(v) / len(qids) for g, v in sorted(groups.items())}
    take = {g: int(x) for g, x in quota.items()}
    for g in sorted(quota, key=lambda g: quota[g] - take[g], reverse=True)[:n - sum(take.values())]:
        take[g] += 1
    out = set()
    for g, v in sorted(groups.items()):
        # sorted(v) hace que el resultado no dependa del orden del archivo.
        out.update(rng.sample(sorted(v), take[g]))
    return out


@cache
def query_labels(domain: str, split: str) -> dict[str, dict[str, str]]:
    """Etiquetas temporales normalizadas por consulta.

    Solo para análisis y oráculos: ``guidance`` no estará disponible en test.

    Args:
        domain: dominio.
        split: ``"train"`` o ``"dev"``.

    Returns:
        ``{qid: {"clase": ..., "intent": ...}}`` con los valores de ``normalize_class``
        y ``normalize_intent``.
    """
    out = {}
    for qid, g in load_guidance(domain, split).items():
        qg = g["query_guidance"]
        out[qid] = {"clase": normalize_class(qg["temporal_reasoning_class_primary"]),
                    "intent": normalize_intent(qg["temporal_intent"])}
    return out
