"""Rutas del proyecto y cargadores de datos del Track 1 (TEMPO).

Copia los cargadores del notebook ``baseline_track1_tempo.ipynb`` para no repetirlos
en cada script, y añade la normalización de las etiquetas temporales de ``guidance``.
"""

import html
import json
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

from bs4 import BeautifulSoup


def find_project_root(start: Path = Path(__file__).resolve()) -> Path:
    """Sube por los directorios hasta encontrar el ``pyproject.toml`` del proyecto."""
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    raise FileNotFoundError("No se encontró pyproject.toml en ningún directorio padre")


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "RETECO-SemEval2027"
T1_DIR = DATA_DIR / "track1_tempo"
EDA_DIR = ROOT / "outputs" / "eda"      # caché del EDA (índices BM25 ya construidos)
EXP_DIR = ROOT / "outputs" / "exp"      # runs y resultados de los experimentos

SEED = 20270101
DOMAINS = sorted(p.name for p in T1_DIR.iterdir() if p.is_dir())


# ----------------------------------------------------------------------------- lectura


def read_jsonl(path: Path) -> list[dict]:
    """Lee un archivo JSON Lines completo (una entrada por línea no vacía)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_documents(domain: str):
    """Recorre el corpus de un dominio sin cargarlo entero: ``{"id", "content"}``."""
    with open(T1_DIR / domain / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@cache
def load_examples(domain: str, split: str) -> list[dict]:
    """Consultas de 1a: ``{id, query, gold_ids, gold_answers}``."""
    return read_jsonl(T1_DIR / domain / f"examples_{split}.jsonl")


@cache
def load_steps(domain: str, split: str) -> list[dict]:
    """Consultas de 1b: ``{id, query, steps: [{step_id, step, step_instruction, gold_ids}]}``."""
    return read_jsonl(T1_DIR / domain / f"steps_{split}.jsonl")


@cache
def load_guidance(domain: str, split: str) -> dict[str, dict]:
    """Metadatos temporales por id de consulta: ``{id: {query_guidance, gold_passage_annotations}}``."""
    return {g["id"]: g for g in read_jsonl(T1_DIR / domain / f"guidance_{split}.jsonl")}


@cache
def load_qrels(domain: str, split: str, subtrack: str = "1a") -> dict[str, dict[str, int]]:
    """Juicios TREC ``{qid: {docid: relevancia}}``; en 1b el qid es el ``step_id``."""
    name = f"qrels_{split}.txt" if subtrack == "1a" else f"qrels_steps_{split}.txt"
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(T1_DIR / domain / name, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qid, _, docid, rel = line.split()
                qrels[qid][docid] = int(rel)
    return dict(qrels)


def parent_query_id(qid: str) -> str:
    """Id de la consulta madre de un ``step_id`` (``"124973_5_step2"`` → ``"124973_5"``)."""
    return re.sub(r"_step\d+$", "", qid)


# ----------------------------------------------------------------------------- limpieza


def split_query(raw: str) -> tuple[str, str]:
    """Separa una consulta StackExchange en ``(título, cuerpo)`` sin HTML.

    Las consultas llegan como ``"<título>\\n<p>cuerpo html</p>..."``.
    """
    title, _, body_html = raw.partition("\n")
    body = BeautifulSoup(body_html, "html.parser").get_text(" ")
    norm = lambda s: re.sub(r"\s+", " ", html.unescape(s)).strip()
    return norm(title), norm(body)


@cache
def clean_query(raw: str) -> str:
    """Consulta como texto plano: ``"título. cuerpo"``."""
    title, body = split_query(raw)
    return f"{title}. {body}" if body else title


# ----------------------------------------------------------------------------- etiquetas temporales

# Clases primarias con al menos ~60 consultas en train; el resto se agrupa en "other".
MAIN_CLASSES = (
    "event_analysis_and_localization",
    "time_period_contextualization",
    "origins_evolution_comparative_analysis",
    "trends_changes_and_cross_period",
    "event_verification_and_authenticity",
    "materials_artifacts_and_provenance",
)
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
INTENT_MAP = {
    "when": "when", "period_definition": "period_definition", "timeline": "timeline",
    "ongoing_status": "ongoing_status", "duration": "duration",
    "order": "order", "before_after": "order", "after": "order", "since": "order",
}


def normalize_class(raw: str) -> str:
    """Clase primaria normalizada: una de ``MAIN_CLASSES`` u ``"other"``."""
    return raw if raw in MAIN_CLASSES else "other"


def normalize_intent(raw: str) -> str:
    """Intención normalizada (``when``, ``duration``, ``order``...) u ``"other"``."""
    return INTENT_MAP.get(raw, "other")


def train_mini_ids(domain: str, split: str = "train", n: int = 40) -> set[str]:
    """Subconjunto ``train-mini`` (PLAN_EXPERIMENTOS.md §1): hasta ``n`` consultas por dominio.

    Muestreo estratificado por clase primaria con reparto proporcional (restos mayores) y
    semilla fija. La clase solo se usa para muestrear, nunca en inferencia.
    """
    import random

    labels = query_labels(domain, split)
    qids = [r["id"] for r in load_examples(domain, split)]
    if len(qids) <= n:
        return set(qids)
    groups: dict[str, list[str]] = defaultdict(list)
    for q in qids:
        groups[labels.get(q, {}).get("clase", "other")].append(q)
    rng = random.Random(f"{SEED}-{domain}-{split}")
    quota = {g: n * len(v) / len(qids) for g, v in sorted(groups.items())}
    take = {g: int(x) for g, x in quota.items()}
    for g in sorted(quota, key=lambda g: quota[g] - take[g], reverse=True)[:n - sum(take.values())]:
        take[g] += 1
    out = set()
    for g, v in sorted(groups.items()):
        out.update(rng.sample(sorted(v), take[g]))
    return out


@cache
def query_labels(domain: str, split: str) -> dict[str, dict[str, str]]:
    """Etiquetas normalizadas por consulta: ``{qid: {"clase": ..., "intent": ...}}``.

    Solo para análisis y oráculos: ``guidance`` no estará disponible en test.
    """
    out = {}
    for qid, g in load_guidance(domain, split).items():
        qg = g["query_guidance"]
        out[qid] = {"clase": normalize_class(qg["temporal_reasoning_class_primary"]),
                    "intent": normalize_intent(qg["temporal_intent"])}
    return out
