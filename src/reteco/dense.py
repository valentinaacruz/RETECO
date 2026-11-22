"""Recuperación densa sobre trozos (chunks) de documento.

Cada documento se parte en trozos de ``CHUNK_TOKENS`` tokens con un solape de
``CHUNK_TOKENS - STRIDE`` y se codifican como mucho ``MAX_CHUNKS`` por documento
(los ~1000 primeros tokens). El score de un documento es el de su mejor trozo
(``agg="max"``) o el de su primer trozo (``agg="first"``, equivale a truncar).

Flujo:

1. ``encode_corpus`` codifica los trozos de un dominio una sola vez y los guarda en
   ``outputs/exp/dense/<modelo>/<dominio>.npz`` (``scripts/encode_corpus.py``).
2. ``DenseIndex`` carga ese archivo, codifica las consultas y calcula el coseno contra
   todos los trozos (búsqueda exacta, sin índice aproximado).

Usa torch: no importarlo junto con ``reteco.bm25`` (JVM de Pyserini).
"""

import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from reteco.data import EXP_DIR, iter_documents
from reteco.fusion import top_k

# Encoders disponibles. Cada modelo espera un prefijo distinto en consultas y documentos
# (lo dicen sus model cards); sin él la calidad baja.
MODELS = {
    "bge-small": {"name": "BAAI/bge-small-en-v1.5",
                  "query_prefix": "Represent this sentence for searching relevant passages: ", "doc_prefix": ""},
    "bge-base": {"name": "BAAI/bge-base-en-v1.5",
                 "query_prefix": "Represent this sentence for searching relevant passages: ", "doc_prefix": ""},
    "e5-base": {"name": "intfloat/e5-base-v2", "query_prefix": "query: ", "doc_prefix": "passage: "},
}
CHUNK_TOKENS = 254      # + [CLS] y [SEP] = 256 posiciones
STRIDE = 224            # 30 tokens de solape entre trozos consecutivos
MAX_CHUNKS = 4          # cubre los ~1000 primeros tokens de cada documento
MAX_CHARS = 12_000      # se corta el texto antes de tokenizar: 4 trozos nunca pasan de aquí
QUERY_MAX_TOKENS = 512  # las consultas largas (mediana ~127 palabras) caben enteras
# Dispositivo: MPS en el Mac, CUDA en el servidor, CPU como último recurso.
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DENSE_DIR = EXP_DIR / "dense"

# Caché de modelos ya cargados en este proceso, por clave de MODELS.
_MODELS: dict[str, SentenceTransformer] = {}


def get_model(key: str) -> SentenceTransformer:
    """Carga (una vez por proceso) el encoder, en fp16 si hay GPU.

    Args:
        key: clave de ``MODELS`` (``"bge-small"``, ``"bge-base"``, ``"e5-base"``).

    Returns:
        El modelo de sentence-transformers en ``DEVICE``.
    """
    if key not in _MODELS:
        model = SentenceTransformer(MODELS[key]["name"], device=DEVICE)
        if DEVICE != "cpu":
            model.half()   # fp16: la mitad de memoria y más rápido; en el EDA no cambió el nDCG
        _MODELS[key] = model
    return _MODELS[key]


def chunk_texts(tokenizer, texts: list[str]) -> tuple[list[str], list[int]]:
    """Parte cada texto en trozos de ``CHUNK_TOKENS`` tokens.

    Los cortes se hacen sobre los offsets de carácter del tokenizador, así que cada
    trozo es un fragmento literal del texto original.

    Args:
        tokenizer: tokenizador rápido (HF) del encoder.
        texts: textos de los documentos.

    Returns:
        ``(trozos, índice del documento de cada trozo)``. Todo documento tiene al
        menos un trozo, aunque esté vacío.
    """
    # Se tokeniza sin tokens especiales ni truncado: solo interesan los offsets de cada token.
    enc = tokenizer([t[:MAX_CHARS] for t in texts], add_special_tokens=False,
                    return_offsets_mapping=True, truncation=False, verbose=False)
    chunks, owner = [], []
    for i, (text, offsets) in enumerate(zip(texts, enc["offset_mapping"])):
        if not offsets:
            # Documento vacío o sin tokens: un único trozo para que el documento exista en el índice.
            chunks.append(text[:MAX_CHARS])
            owner.append(i)
            continue
        # Ventanas que empiezan cada STRIDE tokens, como mucho MAX_CHUNKS.
        for start in range(0, len(offsets), STRIDE)[:MAX_CHUNKS]:
            end = min(start + CHUNK_TOKENS, len(offsets))
            # Del primer carácter del token inicial al último carácter del token final.
            chunks.append(text[offsets[start][0]:offsets[end - 1][1]])
            owner.append(i)
            if end == len(offsets):   # la ventana llegó al final del documento
                break
    return chunks, owner


def encode_corpus(key: str, domain: str, batch_size: int = 64) -> None:
    """Codifica los trozos de un dominio y los guarda en ``outputs/exp/dense/<modelo>/<dominio>.npz``.

    El archivo guarda ``emb`` (fp16, normalizados, en orden de documento),
    ``chunk_doc`` (documento de cada trozo) y ``doc_ids``. Si el archivo ya existe no
    hace nada, así que se puede interrumpir y relanzar por dominios.

    Args:
        key: clave del encoder en ``MODELS``.
        domain: dominio a codificar.
        batch_size: tamaño de batch del encoder.
    """
    path = DENSE_DIR / key / f"{domain}.npz"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = get_model(key), MODELS[key]
    model.max_seq_length = CHUNK_TOKENS + 10  # margen para el prefijo de documento de e5
    t0 = time.time()
    docs = list(iter_documents(domain))
    doc_ids = [d["id"] for d in docs]
    chunks, owner = [], []
    # Se trocea en bloques de 2000 documentos para limitar la memoria del tokenizador.
    for s in range(0, len(docs), 2000):
        c, o = chunk_texts(model.tokenizer, [d["content"] for d in docs[s:s + 2000]])
        chunks.extend(c)
        owner.extend(i + s for i in o)   # índice local del bloque → índice global
    del docs
    print(f"[{key}] {domain}: {len(doc_ids):,} docs → {len(chunks):,} trozos ({time.time() - t0:.0f} s)", flush=True)
    # Ordenar por longitud agrupa textos parecidos en cada batch y reduce el padding.
    order = np.argsort([len(c) for c in chunks])[::-1]
    emb = model.encode([cfg["doc_prefix"] + chunks[i] for i in order], batch_size=batch_size,
                       normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    # Deshace el orden por longitud: la fila order[j] recibe el embedding j.
    out = np.empty((len(chunks), emb.shape[1]), dtype=np.float16)
    out[order] = emb.astype(np.float16)
    np.savez(path, emb=out, chunk_doc=np.array(owner, dtype=np.int32), doc_ids=np.array(doc_ids))
    print(f"[{key}] {domain}: codificado en {(time.time() - t0) / 60:.1f} min", flush=True)


class DenseIndex:
    """Índice denso exacto (producto interno) de un dominio, con agregación por documento.

    Como los embeddings están normalizados, el producto interno es el coseno.

    Attributes:
        key: clave del encoder.
        domain: dominio.
        emb: embeddings de los trozos, ``float32`` (n_trozos × dim).
        doc_ids: id de cada documento, en orden del corpus.
        doc_starts: índice del primer trozo de cada documento en ``emb``.
    """

    def __init__(self, key: str, domain: str):
        """Carga los trozos codificados por ``encode_corpus``.

        Args:
            key: clave del encoder en ``MODELS``.
            domain: dominio.
        """
        z = np.load(DENSE_DIR / key / f"{domain}.npz")
        self.key, self.domain = key, domain
        self.emb = z["emb"].astype(np.float32)   # fp32 para el producto: más preciso y rápido en CPU
        self.doc_ids = list(z["doc_ids"])
        chunk_doc = z["chunk_doc"]
        # Los trozos de un documento son contiguos: doc_starts marca dónde empieza cada uno.
        self.doc_starts = np.flatnonzero(np.r_[True, chunk_doc[1:] != chunk_doc[:-1]])
        assert len(self.doc_starts) == len(self.doc_ids)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Codifica consultas con el prefijo del modelo.

        Args:
            texts: textos de las consultas.

        Returns:
            Embeddings normalizados ``float32`` (n_consultas × dim).
        """
        model = get_model(self.key)
        model.max_seq_length = QUERY_MAX_TOKENS
        prefix = MODELS[self.key]["query_prefix"]
        return model.encode([prefix + t for t in texts], batch_size=32, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def search_vectors(self, qids: list[str], qemb: np.ndarray, k: int = 1000, agg: str = "max") -> dict:
        """Busca a partir de embeddings de consulta ya calculados.

        Args:
            qids: ids de las consultas, en el orden de las filas de ``qemb``.
            qemb: embeddings de las consultas.
            k: documentos por consulta.
            agg: ``"max"`` (score del mejor trozo) o ``"first"`` (score del primer trozo).

        Returns:
            Run ``{qid: {doc_id: score}}``.
        """
        run = {}
        # De 64 en 64 consultas para que la matriz de similitudes quepa en memoria.
        for s in range(0, len(qids), 64):
            sims = qemb[s:s + 64] @ self.emb.T                       # (consultas, trozos)
            if agg == "max":
                # Máximo de cada tramo contiguo de trozos = mejor trozo de cada documento.
                doc_sims = np.maximum.reduceat(sims, self.doc_starts, axis=1)
            else:
                doc_sims = sims[:, self.doc_starts]                  # solo el primer trozo
            for i, qid in enumerate(qids[s:s + 64]):
                run[qid] = top_k(doc_sims[i], self.doc_ids, k)
        return run

    def search_many(self, queries: dict[str, str], k: int = 1000, agg: str = "max") -> dict:
        """Codifica y busca un conjunto de consultas.

        Args:
            queries: ``{qid: texto}``.
            k: documentos por consulta.
            agg: ``"max"`` o ``"first"`` (ver ``search_vectors``).

        Returns:
            Run ``{qid: {doc_id: score}}``.
        """
        qids = list(queries)
        return self.search_vectors(qids, self.encode_queries([queries[q] for q in qids]), k, agg)
