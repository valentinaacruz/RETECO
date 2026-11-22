"""BM25 equivalente al baseline oficial (Lucene analyzer + gensim ``LuceneBM25Model``).

Es la clase ``OfficialStyleBM25`` del notebook ``baseline_track1_tempo.ipynb``, sin
cambios en la fórmula. Reutiliza los índices ya construidos en ``outputs/eda/bm25/``.
Este módulo arranca la JVM de Pyserini: no importarlo junto con ``reteco.dense``.

Fórmula (variante de Lucene, la misma que gensim ``LuceneBM25Model``):

    w(t, d) = idf(t) · tf(t, d) / (tf(t, d) + k1 · (1 − b + b · |d| / avgdl))
    idf(t)  = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))
    score(q, d) = Σ_t w(t, q) · w(t, d)

La consulta también se pondera con BM25 (``w(t, q)``, con ``|q|`` en lugar de ``|d|``),
igual que en el oficial. Ver BASELINE.md, parte 1, para la explicación paso a paso.
"""

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm

from reteco.data import EDA_DIR, iter_documents
from reteco.fusion import top_k

# Analizador de Lucene; se crea la primera vez que se usa porque arrancar la JVM tarda.
_ANALYZER = None


def lucene_analyze(text: str) -> list[str]:
    """Tokeniza con el analizador de Lucene de Pyserini.

    Pasa a minúsculas, quita stopwords en inglés y aplica el stemmer de Porter, como
    el baseline oficial.

    Args:
        text: texto a analizar.

    Returns:
        Lista de términos (con repeticiones, en orden).
    """
    global _ANALYZER
    if _ANALYZER is None:
        # Import tardío: arranca la JVM.
        from pyserini import analysis
        _ANALYZER = analysis.Analyzer(analysis.get_lucene_analyzer())
    return _ANALYZER.analyze(text)


class OfficialStyleBM25:
    """Índice BM25 de un dominio.

    Guarda la matriz de pesos ``w(t, d)`` transpuesta (términos × documentos, CSR): así
    los términos de una consulta son unas pocas filas y el score es un producto
    matriz-vector disperso, mucho más barato en memoria que gensim con los 200 k
    documentos de ``history``.

    Attributes:
        K1, B: parámetros de BM25 del baseline oficial.
        CACHE: directorio de los índices (``outputs/eda/bm25/``).
        domain: dominio del índice.
        doc_ids: ids de documento en el orden del corpus (columnas de ``WT``).
        vocab: ``{término: índice de fila}``.
        idf: idf de cada término (``np.float32``).
        avgdl: longitud media de documento, en términos.
        WT: matriz CSR términos × documentos con los pesos BM25.
    """

    K1, B = 0.9, 0.4
    CACHE = EDA_DIR / "bm25"

    def __init__(self, domain: str):
        """Carga el índice de la caché o lo construye si no existe.

        Args:
            domain: dominio a indexar.

        Side effects:
            Si el índice no existe, escribe ``<dominio>_meta.pkl`` y ``<dominio>_WT.npz``
            en ``CACHE``.
        """
        self.domain = domain
        self.CACHE.mkdir(parents=True, exist_ok=True)
        meta_path, mat_path = self.CACHE / f"{domain}_meta.pkl", self.CACHE / f"{domain}_WT.npz"
        if not (meta_path.exists() and mat_path.exists()):
            self._build(meta_path, mat_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.doc_ids, self.vocab, self.idf, self.avgdl = meta["doc_ids"], meta["vocab"], meta["idf"], meta["avgdl"]
        self.WT = sp.load_npz(mat_path).tocsr()

    def _build(self, meta_path: Path, mat_path: Path) -> None:
        """Analiza el corpus, calcula los pesos BM25 y guarda el índice (pasos 1–4 de BASELINE.md).

        Args:
            meta_path: dónde guardar ``doc_ids``, ``vocab``, ``idf`` y ``avgdl`` (pickle).
            mat_path: dónde guardar la matriz ``WT`` (``scipy.sparse.save_npz``).
        """
        # 1. Matriz de frecuencias documentos × términos en formato CSR, construida a mano
        #    (indptr/indices/data) para no tener todo el corpus tokenizado en memoria.
        vocab, doc_ids = {}, []
        indptr, indices, data = [0], [], []
        for doc in tqdm(iter_documents(self.domain), desc=f"BM25 {self.domain}", leave=False):
            doc_ids.append(doc["id"])
            # setdefault asigna un índice nuevo a cada término la primera vez que aparece.
            tf = Counter(vocab.setdefault(t, len(vocab)) for t in lucene_analyze(doc["content"]))
            indices.extend(tf.keys())
            data.extend(tf.values())
            indptr.append(len(indices))
        tf = sp.csr_matrix((np.array(data, np.float32), np.array(indices, np.int32), np.array(indptr, np.int64)),
                           shape=(len(doc_ids), len(vocab)))
        # 2. Estadísticas del corpus: longitud de cada documento, media y df de cada término.
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(dl.mean())
        df = np.bincount(tf.indices, minlength=tf.shape[1])   # en cuántos documentos sale cada término
        # 3. idf de Lucene (siempre positivo gracias al log1p).
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # 4. Peso BM25 de cada celda no nula. dl_rep repite la longitud de cada documento
        #    tantas veces como términos distintos tiene, para alinearla con tf.data.
        dl_rep = np.repeat(dl, np.diff(tf.indptr))
        tf.data = (idf[tf.indices] * tf.data / (tf.data + self.K1 * (1 - self.B + self.B * dl_rep / avgdl))).astype(np.float32)
        # Se guarda transpuesta (términos × documentos) para leer filas por término al buscar.
        sp.save_npz(mat_path, tf.T.tocsr())
        with open(meta_path, "wb") as f:
            pickle.dump({"doc_ids": doc_ids, "vocab": vocab, "idf": idf, "avgdl": avgdl}, f)

    def query_vector(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Vector BM25 disperso de la consulta.

        La consulta también recibe peso BM25, como en gensim: su longitud ``|q|`` hace el
        papel de ``|d|``. Los términos que no están en el vocabulario se ignoran.

        Args:
            query: texto de la consulta.

        Returns:
            ``(índices de término, pesos)``; dos arrays vacíos si ningún término está en el vocabulario.
        """
        tf = Counter(self.vocab[t] for t in lucene_analyze(query) if t in self.vocab)
        if not tf:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        idx = np.fromiter(tf.keys(), dtype=np.int64)
        f = np.fromiter(tf.values(), dtype=np.float32)
        qlen = f.sum()
        return idx, self.idf[idx] * f / (f + self.K1 * (1 - self.B + self.B * qlen / self.avgdl))

    def scores(self, query: str) -> np.ndarray:
        """Score BM25 de la consulta contra todos los documentos del dominio.

        Args:
            query: texto de la consulta.

        Returns:
            Array de longitud ``len(doc_ids)`` con ``Σ_t w(t, q) · w(t, d)``.
        """
        idx, w = self.query_vector(query)
        if len(idx) == 0:
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        # Filas de los términos de la consulta (términos × docs), traspuestas y por el vector de pesos.
        return np.asarray(self.WT[idx].T @ w).ravel()

    def search(self, query: str, k: int = 1000) -> dict[str, float]:
        """Recupera los mejores documentos de una consulta.

        Args:
            query: texto de la consulta.
            k: número de documentos a devolver.

        Returns:
            ``{doc_id: score}`` de los ``k`` mejores, con el desempate del oficial.
        """
        return top_k(self.scores(query), self.doc_ids, k)

    def search_many(self, queries: dict[str, str], k: int = 1000) -> dict[str, dict[str, float]]:
        """Busca muchas consultas.

        Args:
            queries: ``{qid: texto}``.
            k: documentos por consulta.

        Returns:
            Run ``{qid: {doc_id: score}}``.
        """
        return {qid: self.search(q, k) for qid, q in tqdm(queries.items(), desc=f"bm25:{self.domain}", leave=False)}
