"""Recuperación local de TG-RAG (Algoritmo 1 del paper) adaptada a ranking de documentos.

Para una consulta con embedding ``e_q``, alcance temporal ``T_q`` y candidatos de la
primera etapa (su pool):

1. Aristas candidatas: las de los documentos del pool de la consulta
   (``edge_scope="query"``) o todas las del grafo del dominio (``"pool"``).
2. ``γ_ε = cos(e_q, e_ε)``; las top-``K`` forman ``E_q`` y el subgrafo ``G_q``.
3. Semillas: entidades de las aristas de ``E_q`` con τ ∈ ``T_q``. Si la consulta no
   tiene alcance o ninguna arista cae en él, no se filtra (``fallback``), como hace el
   paper con las consultas sin tiempo (§VI-C).
4. PPR sobre ``G_q`` (``graph_scope="subgraph"``, el paper) o sobre todo el grafo del
   pool (``"pool"``, estilo HippoRAG), con las semillas como personalización → ``s(v)``.
5. ``s(ε) = m(τ, T_q) · (s(v₁) + s(v₂))``, donde ``m`` es el indicador del paper
   (``contained``) o una versión permisiva o suave (ver ``reteco.temporal.temporal_match``).
6. ``s(c) = Π_{ε∈E(c)} (1 + γ_ε) · Σ_{ε∈E(c)} s(ε)``. La ec. (3) dice ``Σ_{ε∈E_q}``,
   pero sería constante entre chunks; se interpreta como ``E(c)``. Se calcula en
   logaritmos para que el producto no desborde.
7. Score del documento: máximo (o suma) sobre sus chunks.

Las ablaciones de la tabla VII del paper son ``PRESETS``. Usa torch vía ``reteco.dense``
(embeddings): no importar junto con ``reteco.bm25``.
"""

import hashlib
from dataclasses import asdict, dataclass, replace

import numpy as np
import scipy.sparse as sp

from reteco.temporal import Scope, temporal_match
from reteco.tgraph import TG_DIR, TemporalGraph


@dataclass(frozen=True)
class TGConfig:
    K: int = 20
    temporal: bool = True
    ppr: bool = True
    static: bool = False
    match: str = "contained"          # contained | overlap | soft
    edge_scope: str = "query"         # query | pool
    graph_scope: str = "subgraph"     # subgraph | pool
    chunk_weight: str = "prod"        # prod | none
    doc_agg: str = "max"              # max | sum
    damping: float = 0.85
    fallback: bool = True

    def name(self, preset: str) -> str:
        """Nombre corto de la variante: solo los parámetros distintos del valor por defecto."""
        default = asdict(replace(TGConfig(), **PRESETS[preset]))
        diff = [f"{k}={v}" for k, v in asdict(self).items() if v != default[k] and k not in PRESETS[preset]]
        return preset + (f"[{','.join(diff)}]" if diff else "")


PRESETS: dict[str, dict] = {
    "full": {"temporal": True, "ppr": True},
    "no_ppr": {"temporal": True, "ppr": False},                    # s(ε) = 1[τ∈T_q]·γ_ε
    "no_temporal": {"temporal": False, "ppr": True},               # PageRank sin personalizar
    "no_temporal_ppr": {"temporal": False, "ppr": False},          # s(ε) = γ_ε
    "static": {"temporal": False, "ppr": False, "static": True},   # sin τ, aristas paralelas fusionadas
}


# ----------------------------------------------------------------------------- embeddings


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def description_embeddings(graph: TemporalGraph, extractor: str, encoder: str, domain: str,
                           batch_size: int = 128) -> np.ndarray:
    """Embeddings normalizados (fp16) de ``graph.descriptions``, con caché incremental.

    La caché ``outputs/exp/tgrag/edge_emb/<extractor>/<encoder>/<dominio>.npz`` guarda
    ``keys`` (hash de cada texto) y ``emb``. Solo se codifican las descripciones nuevas.
    """
    from reteco.dense import MODELS, get_model

    path = TG_DIR / "edge_emb" / extractor / encoder / f"{domain}.npz"
    keys_old, emb_old = np.array([], dtype="<U16"), None
    if path.exists():
        z = np.load(path)
        keys_old, emb_old = z["keys"], z["emb"]
    row = {k: i for i, k in enumerate(keys_old.tolist())}
    keys = [_hash(t) for t in graph.descriptions]
    missing = [i for i, k in enumerate(keys) if k not in row]
    if missing:
        model = get_model(encoder)
        model.max_seq_length = 128
        new = model.encode([MODELS[encoder]["doc_prefix"] + graph.descriptions[i] for i in missing],
                           batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=len(missing) > 50_000).astype(np.float16)
        new_keys = np.array([keys[i] for i in missing], dtype="<U16")
        keys_old = np.concatenate([keys_old, new_keys])
        emb_old = new if emb_old is None else np.vstack([emb_old, new])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=keys_old, emb=emb_old)
        row.update({k: len(row) + j for j, k in enumerate(new_keys.tolist())})
    return emb_old[np.array([row[k] for k in keys], dtype=np.int64)]


def query_embeddings(encoder: str, queries: dict[str, str | list[str]]) -> dict[str, np.ndarray]:
    """Embedding por consulta; si tiene varios textos, la media normalizada."""
    from reteco.dense import MODELS, QUERY_MAX_TOKENS, get_model

    model = get_model(encoder)
    model.max_seq_length = QUERY_MAX_TOKENS
    flat = [(qid, t) for qid, q in queries.items() for t in (q if isinstance(q, list) else [q])]
    emb = model.encode([MODELS[encoder]["query_prefix"] + t for _, t in flat], batch_size=32,
                       normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    acc: dict[str, list[np.ndarray]] = {}
    for (qid, _), e in zip(flat, emb):
        acc.setdefault(qid, []).append(e)
    out = {}
    for qid, es in acc.items():
        v = np.mean(es, axis=0).astype(np.float32)
        out[qid] = v / (np.linalg.norm(v) + 1e-12)
    return out


# ----------------------------------------------------------------------------- PPR


def pagerank(adj: sp.csr_matrix, personalization: np.ndarray | None, damping: float = 0.85,
             iters: int = 100, tol: float = 1e-8) -> np.ndarray:
    """PageRank (personalizado si se da ``personalization``) por iteración de potencias.

    La masa de los nodos sin aristas vuelve a la distribución de personalización.
    """
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0)
    p = np.full(n, 1.0 / n) if personalization is None or personalization.sum() <= 0 \
        else personalization / personalization.sum()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=np.float64), where=deg > 0)
    trans = sp.diags(inv) @ adj                      # fila i: probabilidad de saltar de i a j
    dangling = deg == 0
    r = p.copy()
    for _ in range(iters):
        r_new = damping * (trans.T @ r + r[dangling].sum() * p) + (1 - damping) * p
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


# ----------------------------------------------------------------------------- scoring


def score_query(graph: TemporalGraph, desc_emb: np.ndarray, q: np.ndarray, scopes: list[Scope],
                cand_docs: list[int], cfg: TGConfig) -> tuple[dict[int, float], dict]:
    """Score TG (log s(d)) de los documentos de ``cand_docs`` para una consulta.

    Args:
        graph: grafo del dominio.
        desc_emb: embeddings de ``graph.descriptions`` (fp16, normalizados).
        q: embedding normalizado de la consulta.
        scopes: alcance temporal ``T_q`` (puede estar vacío).
        cand_docs: índices (en ``graph``) de los documentos del pool de la consulta.
        cfg: configuración.

    Returns:
        ``({índice de documento: log score}, diagnóstico)``. Solo aparecen los documentos
        con score positivo; el resto lo ordena la primera etapa al fusionar.
    """
    info = {"aristas_cand": 0, "con_alcance": bool(scopes), "semillas": 0, "fallback": False, "docs_tg": 0}
    edges = np.sort(graph.edges_of(cand_docs)) if cfg.edge_scope == "query" else np.arange(graph.n_edges)
    if len(edges) == 0:
        return {}, info
    uniq, inv = np.unique(graph.desc[edges], return_inverse=True)
    gamma = np.clip(desc_emb[uniq].astype(np.float32) @ q, 0.0, None)[inv]

    if cfg.static:
        # Grafo estático: sin τ, una sola arista por (documento, chunk, e1, e2) → la de mayor γ.
        order = np.argsort(-gamma, kind="stable")
        key = np.stack([graph.doc[edges], graph.chunk[edges], graph.e1[edges], graph.e2[edges]], axis=1)[order]
        _, first = np.unique(key, axis=0, return_index=True)
        keep = np.sort(order[first])
        edges, gamma = edges[keep], gamma[keep]
    info["aristas_cand"] = int(len(edges))

    k = min(cfg.K, len(edges))
    top = np.argpartition(-gamma, k - 1)[:k]
    eq, gq = edges[top], gamma[top]

    def match(idx: np.ndarray) -> np.ndarray:
        return np.array([temporal_match(graph.times[i], scopes, cfg.match) for i in idx], dtype=np.float64)

    use_time = cfg.temporal and not cfg.static and bool(scopes)
    m_q = match(eq) if use_time else np.ones(len(eq))
    if use_time and m_q.sum() == 0 and cfg.fallback:
        use_time, m_q = False, np.ones(len(eq))
        info["fallback"] = True
    if cfg.temporal and not scopes:
        info["fallback"] = cfg.fallback
        if not cfg.fallback:
            return {}, info

    # Aristas que se puntúan: E_q (paper) o todas las de los candidatos (PPR sobre el pool).
    if cfg.graph_scope == "subgraph":
        scored, g_s, m_s = eq, gq, m_q
    else:
        scored = np.intersect1d(graph.edges_of(cand_docs), edges)   # ordenadas, como edges
        g_s = gamma[np.searchsorted(edges, scored)]
        m_s = match(scored) if use_time else np.ones(len(scored))

    if cfg.ppr:
        if cfg.graph_scope == "subgraph":
            nodes, local = np.unique(np.concatenate([graph.e1[eq], graph.e2[eq]]), return_inverse=True)
            l1, l2 = local[:len(eq)], local[len(eq):]
            a = sp.coo_matrix((np.ones(len(eq)), (l1, l2)), shape=(len(nodes), len(nodes))).tocsr()
            adj = (a + a.T).tocsr()
            ent_pos = {int(n): i for i, n in enumerate(nodes)}
        else:
            adj = graph.adjacency()
            ent_pos = None
        n = adj.shape[0]
        pers = None
        if cfg.temporal:
            # Personalización: entidades de las aristas de E_q que caen en T_q (todas si hubo fallback).
            pers = np.zeros(n)
            for e, w in zip(eq, m_q):
                if w > 0:
                    for v in (int(graph.e1[e]), int(graph.e2[e])):
                        pers[ent_pos[v] if ent_pos is not None else v] += w
            info["semillas"] = int((pers > 0).sum())
            if cfg.match != "soft":
                pers = (pers > 0).astype(np.float64)
        rank = pagerank(adj, pers, cfg.damping)
        if ent_pos is not None:
            sv1 = np.array([rank[ent_pos[int(v)]] for v in graph.e1[scored]])
            sv2 = np.array([rank[ent_pos[int(v)]] for v in graph.e2[scored]])
        else:
            sv1, sv2 = rank[graph.e1[scored]], rank[graph.e2[scored]]
        s_edge = m_s * (sv1 + sv2)
    else:
        s_edge = m_s * g_s

    # Chunks: Σ s(ε) y Σ log(1 + γ) por (documento, chunk); documento = max (o suma) de sus chunks.
    acc: dict[tuple[int, int], list[float]] = {}
    for e, s, g in zip(scored, s_edge, g_s):
        a = acc.setdefault((int(graph.doc[e]), int(graph.chunk[e])), [0.0, 0.0])
        a[0] += s
        a[1] += np.log1p(g)
    doc_scores: dict[int, float] = {}
    for (d, _), (s_sum, log_w) in acc.items():
        if s_sum <= 0:
            continue
        val = np.log(s_sum) + (log_w if cfg.chunk_weight == "prod" else 0.0)
        if d not in doc_scores:
            doc_scores[d] = val
        elif cfg.doc_agg == "max":
            doc_scores[d] = max(doc_scores[d], val)
        else:
            doc_scores[d] = float(np.logaddexp(doc_scores[d], val))
    info["docs_tg"] = len(doc_scores)
    return doc_scores, info
