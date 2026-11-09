"""BM25 equivalente al baseline oficial (Lucene analyzer + gensim ``LuceneBM25Model``).

Es la clase ``OfficialStyleBM25`` del notebook ``baseline_track1_tempo.ipynb``, sin
cambios en la fórmula. Reutiliza los índices ya construidos en ``outputs/eda/bm25/``.
Este módulo arranca la JVM de Pyserini: no importarlo junto con ``reteco.dense``.
"""

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm

from reteco.data import EDA_DIR, iter_documents
from reteco.fusion import top_k

_ANALYZER = None


def lucene_analyze(text: str) -> list[str]:
    """Tokeniza con el analizador de Lucene de Pyserini (minúsculas, sin stopwords, Porter)."""
    global _ANALYZER
    if _ANALYZER is None:
        from pyserini import analysis
        _ANALYZER = analysis.Analyzer(analysis.get_lucene_analyzer())
    return _ANALYZER.analyze(text)


class OfficialStyleBM25:
    """Índice BM25 de un dominio con la matriz de pesos transpuesta (términos × documentos, CSR)."""

    K1, B = 0.9, 0.4
    CACHE = EDA_DIR / "bm25"

    def __init__(self, domain: str):
        """Carga el índice de la caché o lo construye si no existe."""
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
        """Analiza el corpus, calcula los pesos BM25 y guarda el índice (pasos 1–4 de BASELINE.md)."""
        vocab, doc_ids = {}, []
        indptr, indices, data = [0], [], []
        for doc in tqdm(iter_documents(self.domain), desc=f"BM25 {self.domain}", leave=False):
            doc_ids.append(doc["id"])
            tf = Counter(vocab.setdefault(t, len(vocab)) for t in lucene_analyze(doc["content"]))
            indices.extend(tf.keys())
            data.extend(tf.values())
            indptr.append(len(indices))
        tf = sp.csr_matrix((np.array(data, np.float32), np.array(indices, np.int32), np.array(indptr, np.int64)),
                           shape=(len(doc_ids), len(vocab)))
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(dl.mean())
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl_rep = np.repeat(dl, np.diff(tf.indptr))
        tf.data = (idf[tf.indices] * tf.data / (tf.data + self.K1 * (1 - self.B + self.B * dl_rep / avgdl))).astype(np.float32)
        sp.save_npz(mat_path, tf.T.tocsr())
        with open(meta_path, "wb") as f:
            pickle.dump({"doc_ids": doc_ids, "vocab": vocab, "idf": idf, "avgdl": avgdl}, f)

    def query_vector(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """(índices de término, pesos BM25) de la consulta; la consulta también recibe peso BM25, como en gensim."""
        tf = Counter(self.vocab[t] for t in lucene_analyze(query) if t in self.vocab)
        if not tf:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        idx = np.fromiter(tf.keys(), dtype=np.int64)
        f = np.fromiter(tf.values(), dtype=np.float32)
        qlen = f.sum()
        return idx, self.idf[idx] * f / (f + self.K1 * (1 - self.B + self.B * qlen / self.avgdl))

    def scores(self, query: str) -> np.ndarray:
        """Score BM25 de la consulta contra todos los documentos del dominio."""
        idx, w = self.query_vector(query)
        if len(idx) == 0:
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        return np.asarray(self.WT[idx].T @ w).ravel()

    def search(self, query: str, k: int = 1000) -> dict[str, float]:
        """Recupera los ``k`` mejores documentos ``{doc_id: score}``."""
        return top_k(self.scores(query), self.doc_ids, k)

    def search_many(self, queries: dict[str, str], k: int = 1000) -> dict[str, dict[str, float]]:
        """Run ``{qid: {doc_id: score}}`` para un diccionario ``{qid: texto}``."""
        return {qid: self.search(q, k) for qid, q in tqdm(queries.items(), desc=f"bm25:{self.domain}", leave=False)}
