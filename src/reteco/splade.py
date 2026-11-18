"""Recuperación *sparse* aprendida (SPLADE) sobre trozos de documento (``PLAN_SPLADE.md``).

SPLADE pasa el texto por un BERT con su cabeza de *masked language modeling* y obtiene,
para cada término del vocabulario (30,522 en BERT *uncased*), el peso
``max_tokens log(1 + ReLU(logit))``. El resultado es un vector disperso con unos cientos
de términos activos, que incluyen términos que no aparecen en el texto (expansión). El
score consulta–documento es el producto escalar de sus vectores.

Este módulo replica el diseño de ``reteco.dense`` para que ``scripts/run_experiment.py``,
``scripts/fuse_runs.py`` y ``reteco.evaluate`` sirvan sin cambios:

1. ``encode_corpus`` trocea cada documento igual que el denso (mismos ``chunk_texts`` y
   constantes), codifica los trozos y guarda una matriz CSR (trozos × vocabulario) en
   ``outputs/exp/splade/<modelo>/<dominio>.npz`` (``scripts/encode_splade.py``).
2. ``SpladeIndex`` carga esa matriz, guarda su transpuesta (vocabulario × trozos, un
   índice invertido), codifica las consultas y busca con un producto disperso exacto.

Diferencias con el repo de IIMAS-RAG (``PLAN_SPLADE.md`` §2.2): aquí la salida del modelo
se convierte a disperso bloque a bloque (nunca se materializa la matriz densa
trozos × 30,522), se codifica en fp16 y se trocean los documentos.

Usa torch: no importarlo junto con ``reteco.bm25`` (JVM de Pyserini).
"""

import re
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sentence_transformers import SparseEncoder

from reteco.data import EXP_DIR, iter_documents
from reteco.dense import CHUNK_TOKENS, DENSE_DIR, DEVICE, QUERY_MAX_TOKENS, chunk_texts
from reteco.fusion import top_k

# Modelos disponibles: clave corta → modelo de Hugging Face (PLAN_SPLADE.md §3.1).
MODELS = {
    # Réplica de IIMAS-RAG (SPLADE++ "EnsembleDistil"). Descarga libre.
    "splade-ed": "naver/splade-cocondenser-ensembledistil",
    # Más reciente. Restringido (gated): hay que aceptar los términos en HF y tener token.
    "splade-v3": "naver/splade-v3",
    # Versión DistilBERT de splade-v3: más rápida. Descarga libre.
    "splade-v3-db": "naver/splade-v3-distilbert",
    # Alternativa con licencia Apache 2.0 (los de NAVER son CC BY-NC-SA 4.0).
    "os-v2": "opensearch-project/opensearch-neural-sparse-encoding-v2-distill",
}
# Formas de pasar del score de cada trozo al del documento (ver SpladeIndex.search_vectors).
AGGS = ("max", "first", "maxpool")
SPLADE_DIR = EXP_DIR / "splade"
# Trozos por bloque de codificación. Cada bloque se convierte a CSR en cuanto sale del
# modelo, así que la memoria la marca la CSR, no un tensor denso de 30,522 columnas.
BLOCK = 20_000
# Consultas por lote en la búsqueda: la matriz densa de scores (64 × trozos del dominio)
# ocupa ~85 MB en history, el dominio más grande.
QUERY_BATCH = 64

# Caché de modelos ya cargados en este proceso, por clave de MODELS.
_MODELS: dict[str, SparseEncoder] = {}

# Nombre de retriever: <modelo>[-q<N>][-noexp][@<agregación>], p. ej. "splade-ed-q64@maxpool".
# La clave del modelo es no codiciosa (+?) para que "-q64" y "-noexp" no se la coman, aunque
# las claves tengan guiones ("splade-v3-db").
_NAME = re.compile(r"^(?P<key>.+?)(?:-q(?P<q>\d+))?(?P<noexp>-noexp)?(?:@(?P<agg>\w+))?$")


# ----------------------------------------------------------------------------- nombres

def parse_retriever(name: str) -> dict | None:
    """Interpreta un nombre de retriever SPLADE.

    Gramática: ``<modelo>[-q<N>][-noexp][@<agregación>]``.

    - ``-q<N>``: poda la consulta a sus N términos de más peso (SP3).
    - ``-noexp``: anula la expansión de la consulta; solo cuentan sus propios tokens (SP5).
    - ``@max`` (por defecto), ``@first`` o ``@maxpool``: agregación por documento.

    Args:
        name: nombre del retriever, p. ej. ``"splade-ed"`` o ``"splade-v3-q64@first"``.

    Returns:
        ``{"key", "q_max_dims", "noexp", "agg"}``, o ``None`` si el nombre no es de un
        modelo de ``MODELS`` (entonces es BM25 o un denso).

    Raises:
        ValueError: si el modelo es SPLADE pero la agregación o la poda no son válidas.
    """
    m = _NAME.match(name)
    if not m or m["key"] not in MODELS:
        return None
    agg = m["agg"] or "max"
    if agg not in AGGS:
        raise ValueError(f"agregación desconocida {agg!r} en {name!r}; opciones: {AGGS}")
    q = int(m["q"]) if m["q"] else None
    if q is not None and q <= 0:
        raise ValueError(f"la poda de la consulta debe ser positiva: {name!r}")
    return {"key": m["key"], "q_max_dims": q, "noexp": bool(m["noexp"]), "agg": agg}


# ----------------------------------------------------------------------------- modelo y codificación

def get_model(key: str) -> SparseEncoder:
    """Carga (una vez por proceso) el modelo SPLADE, en fp16 si hay GPU.

    Args:
        key: clave de ``MODELS``.

    Returns:
        El ``SparseEncoder`` en ``DEVICE``.
    """
    if key not in _MODELS:
        model = SparseEncoder(MODELS[key], device=DEVICE)
        if DEVICE != "cpu":
            # fp16: la mitad de memoria para los logits (batch × tokens × 30,522), que son el
            # tensor más grande, y más velocidad. El repo de IIMAS codificaba en fp32.
            model.half()
        if model.max_active_dims is not None:
            # Si la configuración del modelo trae un límite de términos activos, ST lo aplica
            # en silencio a consultas y documentos. Se avisa para que quede en el log.
            print(f"[{key}] aviso: el modelo poda a {model.max_active_dims} términos activos", flush=True)
        _MODELS[key] = model
    return _MODELS[key]


def sparse_to_csr(t: torch.Tensor) -> sp.csr_matrix:
    """Convierte un tensor disperso COO de torch (n × V) en una CSR de scipy en float32.

    Args:
        t: salida de ``SparseEncoder.encode_*`` con ``convert_to_sparse_tensor=True``.

    Returns:
        La misma matriz en CSR (float32; scipy no opera en float16).
    """
    t = t.coalesce()
    idx = t.indices().numpy()
    vals = t.values().float().numpy()
    return sp.csr_matrix((vals, (idx[0], idx[1])), shape=tuple(t.shape))


def encode_texts(model: SparseEncoder, texts: list[str], batch_size: int = 32, block: int = BLOCK,
                 kind: str = "document", log_prefix: str | None = None) -> sp.csr_matrix:
    """Codifica textos por bloques y devuelve sus vectores SPLADE como CSR (n × V).

    Los textos se ordenan de más largo a más corto antes de partirlos en bloques. Así el
    primer bloque es el más pesado: si el batch no cabe en la GPU, el error sale en los
    primeros minutos y no a mitad de un dominio. Al final se deshace el orden.

    Args:
        model: modelo de ``get_model`` (o cualquier ``SparseEncoder``).
        texts: textos a codificar.
        batch_size: tamaño de batch del modelo.
        block: textos por bloque; cada bloque pasa a CSR en cuanto sale del modelo.
        kind: ``"document"`` o ``"query"`` (usa ``encode_document`` o ``encode_query``, que
            aplican el prompt o la ruta de cada tipo si el modelo los define).
        log_prefix: si se da, imprime el avance de cada bloque con ese prefijo.

    Returns:
        CSR float32 con una fila por texto, en el orden de ``texts``.
    """
    encode = model.encode_document if kind == "document" else model.encode_query
    # Orden por longitud en caracteres, de mayor a menor (aproxima la longitud en tokens).
    order = np.argsort([-len(t) for t in texts], kind="stable")
    parts, t0 = [], time.time()
    for s in range(0, len(texts), block):
        batch = [texts[i] for i in order[s:s + block]]
        # max_active_dims=None: sin poda (salvo que el modelo la traiga en su configuración,
        # ver get_model). save_to_cpu: cada batch sale de la GPU en cuanto se calcula.
        emb = encode(batch, batch_size=batch_size, convert_to_tensor=True,
                     convert_to_sparse_tensor=True, save_to_cpu=True, show_progress_bar=False)
        parts.append(sparse_to_csr(emb))
        if log_prefix:
            done = min(s + block, len(texts))
            rate = done / (time.time() - t0)
            print(f"{log_prefix} {done:,}/{len(texts):,} trozos · {rate:,.0f}/s · "
                  f"faltan ~{(len(texts) - done) / rate / 60:.0f} min", flush=True)
    if not parts:
        return sp.csr_matrix((0, model.transformers_model.config.vocab_size), dtype=np.float32)
    sorted_mat = sp.vstack(parts, format="csr")
    # La fila j de sorted_mat es el texto order[j]; argsort(order) da, para cada texto,
    # la fila en la que quedó.
    return sorted_mat[np.argsort(order)]


def prune_rows(mat: sp.csr_matrix, n: int) -> sp.csr_matrix:
    """Deja en cada fila solo sus ``n`` valores más altos (poda de la consulta, SP3).

    Hace lo mismo que ``max_active_dims`` de sentence-transformers (todos los pesos SPLADE
    son ≥ 0), pero después de cualquier máscara: con ``-noexp`` primero se quitan las
    expansiones y luego se poda, y no al revés.

    Args:
        mat: CSR (consultas × V).
        n: términos que se conservan por fila.

    Returns:
        Una CSR nueva con como mucho ``n`` valores por fila.
    """
    mat = mat.tocsr(copy=True)
    for i in range(mat.shape[0]):
        lo, hi = mat.indptr[i], mat.indptr[i + 1]
        if hi - lo > n:
            row = mat.data[lo:hi]
            # Los que no están entre los n mayores pasan a 0 y se eliminan al final.
            row[np.argpartition(-row, n - 1)[n:]] = 0.0
    mat.eliminate_zeros()
    return mat


def query_token_mask(tokenizer, texts: list[str], n_cols: int) -> sp.csr_matrix:
    """Máscara 0/1 (consultas × V) con los tokens que aparecen en cada consulta.

    Sirve para la ablación ``-noexp`` (SP5): al multiplicar el vector SPLADE de la consulta
    por esta máscara solo quedan los términos que están en su texto, con el peso aprendido.

    Args:
        tokenizer: tokenizador del modelo SPLADE.
        texts: textos de las consultas.
        n_cols: tamaño del vocabulario de salida del modelo.

    Returns:
        CSR float32 con un 1 en cada (consulta, token de la consulta).
    """
    # Misma truncación que al codificar la consulta; sin [CLS]/[SEP], que no son texto.
    ids = tokenizer(texts, add_special_tokens=False, truncation=True,
                    max_length=QUERY_MAX_TOKENS)["input_ids"]
    rows = np.repeat(np.arange(len(ids)), [len(x) for x in ids])
    cols = np.concatenate([np.asarray(x, dtype=np.int64) for x in ids]) if ids else np.array([], np.int64)
    mask = sp.csr_matrix((np.ones(len(cols), np.float32), (rows, cols)), shape=(len(texts), n_cols))
    mask.data[:] = 1.0   # los tokens repetidos se suman al construir la CSR; aquí vuelven a 1
    return mask


# ----------------------------------------------------------------------------- corpus en disco

def save_matrix(path: Path, mat: sp.csr_matrix, chunk_doc, doc_ids) -> None:
    """Guarda la matriz de un dominio en ``.npz`` de forma atómica.

    Se escribe primero ``<dominio>.tmp.npz`` y después se renombra. Así un proceso
    interrumpido nunca deja un archivo a medias que ``encode_corpus`` daría por bueno.

    Args:
        path: destino (``.../<dominio>.npz``).
        mat: CSR (trozos × V); los valores se guardan en float16 (sobra precisión para
            pesos de 0 a ~3 y el índice ocupa un tercio menos).
        chunk_doc: índice del documento de cada trozo.
        doc_ids: id de cada documento, en orden del corpus.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")   # termina en .npz: np.savez no añade otra extensión
    np.savez(tmp, data=mat.data.astype(np.float16), indices=mat.indices.astype(np.int32),
             indptr=mat.indptr.astype(np.int64), shape=np.array(mat.shape, dtype=np.int64),
             chunk_doc=np.asarray(chunk_doc, dtype=np.int32), doc_ids=np.asarray(doc_ids))
    tmp.replace(path)


def load_matrix(path: Path) -> tuple[sp.csr_matrix, np.ndarray, list[str]]:
    """Lee un archivo de ``save_matrix``.

    Args:
        path: archivo ``.npz``.

    Returns:
        ``(CSR float32 trozos × V, chunk_doc, doc_ids)``.
    """
    z = np.load(path)
    mat = sp.csr_matrix((z["data"].astype(np.float32), z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return mat, z["chunk_doc"], list(z["doc_ids"])


def chunk_domain(tokenizer, domain: str) -> tuple[list[str], list[str], list[int]]:
    """Trocea los documentos de un dominio exactamente como ``reteco.dense.encode_corpus``.

    Args:
        tokenizer: tokenizador rápido del modelo SPLADE.
        domain: dominio.

    Returns:
        ``(doc_ids, trozos, documento de cada trozo)``.
    """
    docs = list(iter_documents(domain))
    doc_ids = [d["id"] for d in docs]
    chunks, owner = [], []
    # Bloques de 2000 documentos, como en el denso, para limitar la memoria del tokenizador.
    for s in range(0, len(docs), 2000):
        c, o = chunk_texts(tokenizer, [d["content"] for d in docs[s:s + 2000]])
        chunks.extend(c)
        owner.extend(i + s for i in o)   # índice local del bloque → índice global
    return doc_ids, chunks, owner


def encode_corpus(key: str, domain: str, batch_size: int = 32, block: int = BLOCK) -> None:
    """Codifica los trozos de un dominio y los guarda en ``outputs/exp/splade/<modelo>/<dominio>.npz``.

    Si el archivo ya existe no hace nada, así que se puede interrumpir y relanzar por
    dominios. Imprime el número de trozos, si coinciden con los del denso ``bge-base``,
    los términos activos por trozo y el tiempo (las medidas que pide SP0).

    Args:
        key: clave de ``MODELS``.
        domain: dominio.
        batch_size: tamaño de batch del modelo.
        block: trozos por bloque de codificación.
    """
    path = SPLADE_DIR / key / f"{domain}.npz"
    if path.exists():
        return
    model = get_model(key)
    model.max_seq_length = CHUNK_TOKENS + 10   # el mismo margen que el denso
    t0 = time.time()
    doc_ids, chunks, owner = chunk_domain(model.tokenizer, domain)
    print(f"[{key}] {domain}: {len(doc_ids):,} docs → {len(chunks):,} trozos ({time.time() - t0:.0f} s)", flush=True)

    # Si el tokenizador tiene el mismo vocabulario que bge-base, los trozos son los mismos
    # que los del denso y "max"/"first" se comparan uno a uno (PLAN_SPLADE.md §3.2).
    ref = DENSE_DIR / "bge-base" / f"{domain}.npz"
    if ref.exists():
        same = np.array_equal(np.load(ref)["chunk_doc"], np.asarray(owner, dtype=np.int32))
        print(f"[{key}] {domain}: trozos iguales a bge-base: {'sí' if same else 'NO'}", flush=True)

    mat = encode_texts(model, chunks, batch_size, block, "document", log_prefix=f"[{key}] {domain}:")
    nnz = np.diff(mat.indptr)
    save_matrix(path, mat, owner, doc_ids)
    print(f"[{key}] {domain}: términos activos por trozo: media {nnz.mean():.0f}, "
          f"p50 {np.median(nnz):.0f}, p99 {np.percentile(nnz, 99):.0f} · "
          f"{path.stat().st_size / 2**30:.2f} GB · {(time.time() - t0) / 60:.1f} min", flush=True)


# ----------------------------------------------------------------------------- búsqueda

class SpladeIndex:
    """Índice SPLADE exacto de un dominio, con agregación por documento.

    Attributes:
        key: clave del modelo.
        domain: dominio.
        q_max_dims: poda de la consulta (``None`` = sin poda).
        noexp: si se anula la expansión de la consulta.
        agg: agregación por defecto (``run_experiment.open_index`` la fija según el nombre).
        D: CSR float32 trozos × V.
        DT: transpuesta de ``D`` en CSR (V × trozos): cada fila es la lista de trozos en
            que aparece un término, es decir, un índice invertido.
        doc_ids: id de cada documento, en orden del corpus.
        doc_starts: índice del primer trozo de cada documento.
    """

    def __init__(self, key: str, domain: str, q_max_dims: int | None = None, noexp: bool = False,
                 path: Path | None = None):
        """Carga la matriz de ``encode_corpus``.

        Args:
            key: clave de ``MODELS`` (o cualquier clave de ``_MODELS`` en las pruebas).
            domain: dominio.
            q_max_dims: si se da, cada consulta se poda a sus N términos de más peso.
            noexp: si es True, solo cuentan los términos que aparecen en la consulta.
            path: archivo a cargar (por defecto ``outputs/exp/splade/<modelo>/<dominio>.npz``).
        """
        self.key, self.domain, self.q_max_dims, self.noexp = key, domain, q_max_dims, noexp
        self.agg = "max"
        self.D, chunk_doc, self.doc_ids = load_matrix(path or SPLADE_DIR / key / f"{domain}.npz")
        self.DT = self.D.T.tocsr()
        # Los trozos de un documento son contiguos: doc_starts marca dónde empieza cada uno.
        self.doc_starts = np.flatnonzero(np.r_[True, chunk_doc[1:] != chunk_doc[:-1]])
        assert len(self.doc_starts) == len(self.doc_ids)
        self._doc_T = None   # transpuesta de la matriz por documento (maxpool); se crea al usarla

    def doc_matrix_T(self) -> sp.csr_matrix:
        """Transpuesta (V × documentos) de los vectores ``maxpool`` de cada documento.

        El vector ``maxpool`` de un documento es el máximo, término a término, de los vectores
        de sus trozos: un único vector SPLADE que cubre sus ~1000 primeros tokens. Se
        construye una vez, a partir de la misma codificación.

        Returns:
            CSR float32 (V × documentos).
        """
        if self._doc_T is None:
            n_docs, n_chunks = len(self.doc_starts), self.D.shape[0]
            counts = np.diff(np.r_[self.doc_starts, n_chunks])   # trozos por documento
            mat = self.D[self.doc_starts]                          # primer trozo de cada documento
            for j in range(1, counts.max()):
                docs = np.flatnonzero(counts > j)                  # documentos con un trozo j
                # Matriz de selección (documentos × trozos) con un 1 en (d, j-ésimo trozo de d):
                # al multiplicarla por D da, en la fila d, el trozo j de d (o una fila vacía).
                sel = sp.csr_matrix((np.ones(len(docs), np.float32), (docs, self.doc_starts[docs] + j)),
                                    shape=(n_docs, n_chunks))
                # Los pesos son ≥ 0, así que el máximo con un cero implícito no cambia nada.
                mat = mat.maximum(sel @ self.D)
            self._doc_T = mat.T.tocsr()
        return self._doc_T

    def encode_queries(self, texts: list[str]) -> sp.csr_matrix:
        """Codifica consultas, con la máscara ``noexp`` y la poda si se pidieron.

        Args:
            texts: textos de las consultas.

        Returns:
            CSR float32 (consultas × V).
        """
        model = get_model(self.key)
        model.max_seq_length = QUERY_MAX_TOKENS   # las consultas largas caben enteras (mediana ~127 palabras)
        q = encode_texts(model, texts, batch_size=32, kind="query")
        if self.noexp:
            q = q.multiply(query_token_mask(model.tokenizer, texts, q.shape[1])).tocsr()
            q.eliminate_zeros()
        if self.q_max_dims:
            q = prune_rows(q, self.q_max_dims)
        return q

    def search_vectors(self, qids: list[str], qmat: sp.csr_matrix, k: int = 1000, agg: str = "max") -> dict:
        """Busca a partir de vectores de consulta ya calculados.

        Args:
            qids: ids de las consultas, en el orden de las filas de ``qmat``.
            qmat: CSR (consultas × V).
            k: documentos por consulta.
            agg: ``"max"`` (score del mejor trozo), ``"first"`` (del primer trozo) o
                ``"maxpool"`` (score del vector máximo de los trozos de cada documento).

        Returns:
            Run ``{qid: {doc_id: score}}``.
        """
        if agg not in AGGS:
            raise ValueError(f"agregación desconocida {agg!r}; opciones: {AGGS}")
        index_T = self.doc_matrix_T() if agg == "maxpool" else self.DT
        run = {}
        for s in range(0, len(qids), QUERY_BATCH):
            # Disperso × disperso: para cada término de la consulta se recorre solo su lista de
            # trozos (fila de index_T). Después se pasa a denso para agregar y ordenar.
            sims = (qmat[s:s + QUERY_BATCH] @ index_T).toarray()      # (consultas, trozos o docs)
            if agg == "max":
                # Máximo de cada tramo contiguo de trozos = mejor trozo de cada documento.
                doc_sims = np.maximum.reduceat(sims, self.doc_starts, axis=1)
            elif agg == "first":
                doc_sims = sims[:, self.doc_starts]
            else:
                doc_sims = sims                                        # ya es una columna por documento
            for i, qid in enumerate(qids[s:s + QUERY_BATCH]):
                run[qid] = top_k(doc_sims[i], self.doc_ids, k)
        return run

    def search_many(self, queries: dict[str, str], k: int = 1000, agg: str = "max") -> dict:
        """Codifica y busca un conjunto de consultas (misma interfaz que ``DenseIndex``).

        Args:
            queries: ``{qid: texto}``.
            k: documentos por consulta.
            agg: ver ``search_vectors``.

        Returns:
            Run ``{qid: {doc_id: score}}``.
        """
        qids = list(queries)
        return self.search_vectors(qids, self.encode_queries([queries[q] for q in qids]), k, agg)
