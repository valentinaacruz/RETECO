"""Utilidades de ranking sin dependencias pesadas: top-k y fusión de runs.

Un *run* es siempre un diccionario ``{qid: {doc_id: score}}``. Este módulo no importa
ni la JVM ni torch, así que lo pueden usar todos los demás.
"""

from collections import defaultdict

import numpy as np


def top_k(scores: np.ndarray, ids: list[str], k: int) -> dict[str, float]:
    """Los ``k`` mejores documentos; los empates se resuelven por orden en el corpus.

    Replica el ``sorted`` estable del baseline oficial: ``lexsort`` ordena por la
    última clave primero (score descendente) y desempata por posición.

    Args:
        scores: score de cada documento, en el orden del corpus.
        ids: id de cada documento, en el mismo orden.
        k: cuántos devolver (se recorta a ``len(scores)``).

    Returns:
        ``{doc_id: score}`` con los ``k`` mejores, insertados en orden de ranking.
    """
    k = min(k, len(scores))
    # argpartition encuentra los k mayores en O(n) sin ordenar todo el corpus...
    cand = np.argpartition(-scores, k - 1)[:k]
    # ...y solo esos k se ordenan: por -score y, en empate, por posición en el corpus.
    cand = cand[np.lexsort((cand, -scores[cand]))]
    return {ids[i]: float(scores[i]) for i in cand}


def rrf_fuse(runs: list[dict], k: int = 60, depth: int = 1000, weights: list[float] | None = None) -> dict:
    """Reciprocal Rank Fusion: ``score(d) = Σ_i w_i / (k + rank_i(d))``.

    Solo usa la posición de cada documento en cada run, no su score, así que se pueden
    fusionar runs con escalas distintas (BM25 y coseno, por ejemplo).

    Args:
        runs: runs ``{qid: {doc_id: score}}``; se fusionan las consultas del primero.
        k: constante de suavizado (60 es el valor habitual).
        depth: documentos que se conservan por consulta.
        weights: peso de cada run (por defecto, todos 1).

    Returns:
        Run fusionado ``{qid: {doc_id: score_rrf}}``, ordenado por score.
    """
    weights = weights or [1.0] * len(runs)
    fused = {}
    for qid in runs[0]:
        acc = defaultdict(float)
        for run, w in zip(runs, weights):
            # Un documento ausente de un run no suma nada por ese run.
            ranked = sorted(run.get(qid, {}).items(), key=lambda x: -x[1])
            for r, (d, _) in enumerate(ranked, start=1):   # rank empieza en 1
                acc[d] += w / (k + r)
        fused[qid] = dict(sorted(acc.items(), key=lambda x: -x[1])[:depth])
    return fused


def rrf_multi(run_lists: dict[str, list[dict[str, float]]], k: int = 60, depth: int = 1000) -> dict:
    """RRF por consulta cuando cada consulta tiene varios rankings (uno por texto de búsqueda).

    Se usa en variantes como ``rrf(limpia,titulo)``: la misma consulta se busca con dos
    textos y sus dos rankings se fusionan.

    Args:
        run_lists: ``{qid: [ranking_1, ranking_2, ...]}``, cada ranking ``{doc_id: score}``.
        k: constante de RRF.
        depth: documentos que se conservan por consulta.

    Returns:
        Run fusionado ``{qid: {doc_id: score_rrf}}``.
    """
    return {qid: rrf_fuse([{qid: r} for r in rankings], k, depth)[qid] for qid, rankings in run_lists.items()}
