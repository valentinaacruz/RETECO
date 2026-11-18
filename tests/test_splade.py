"""Pruebas de ``reteco.splade`` (PLAN_SPLADE.md §3.4).

Casi todas usan un BERT diminuto con pesos aleatorios, creado en un directorio temporal:
no descargan nada y tardan segundos. Comprueban la tubería, no la calidad del modelo:

1. la conversión a CSR por bloques da los mismos vectores que la fórmula SPLADE escrita
   a mano (``log1p(relu(logits)) · máscara``, máximo sobre los tokens), en el orden original;
2. las agregaciones ``max``, ``first`` y ``maxpool`` coinciden con un cálculo por fuerza bruta;
3. los empates se resuelven por orden en el corpus, como en el baseline oficial;
4. la poda de la consulta (``-qN``) y la ablación ``-noexp`` hacen lo que dicen;
5. el nombre de los retrievers se interpreta bien.

La prueba con el modelo real (``splade-ed``, ~440 MB) solo corre si se pide:

    .venv/bin/python tests/test_splade.py
    RETECO_SPLADE_REAL=1 .venv/bin/python tests/test_splade.py

(también funciona con ``pytest tests/`` si se instala).
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

# Permite importar el paquete reteco (en src/) sin instalarlo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentence_transformers import SparseEncoder  # noqa: E402
from sentence_transformers.sparse_encoder.modules import SpladePooling, Transformer  # noqa: E402

from reteco import splade  # noqa: E402
from reteco.dense import chunk_texts  # noqa: E402
from reteco.fusion import top_k  # noqa: E402

# Directorio temporal que vive lo que dure el proceso (modelo diminuto e índices de juguete).
_TMP = tempfile.TemporaryDirectory()
TMP = Path(_TMP.name)

# Vocabulario del modelo diminuto: palabras de los textos de prueba (lo demás será [UNK]).
WORDS = ["the", "bitcoin", "halving", "in", "2020", "price", "was", "high", "before", "after", "war",
         "treaty", "signed", "king", "century", "records", "census", "parish", "museum", "coin", "silver",
         "minted", "when", "did", "start", "end", "of", "a", "an", "and", "is", "it", "true", "that", "1834",
         "reform", "law", "court", "ruling", "who", "first", "used", "term", "origin"]
TEXTS = [
    "The bitcoin halving in 2020",
    "price was high before the halving",
    "",                                                     # texto vacío: solo [CLS] y [SEP]
    "war treaty signed by the king in the century",
    "parish records and census records",
    "museum coin silver minted",
    "when did the war start and when did it end",
    "is it true that the 1834 reform law was signed",
    "court ruling",
    "who first used the term and what is its origin zzz qqq",   # palabras fuera del vocabulario
]


class Skip(Exception):
    """Prueba omitida a propósito (p. ej. la del modelo real sin ``RETECO_SPLADE_REAL=1``)."""


_TINY: dict = {}


def tiny() -> tuple[SparseEncoder, BertForMaskedLM, BertTokenizerFast]:
    """Crea (una vez) un SPLADE diminuto y aleatorio: BERT de 2 capas + cabeza MLM + SpladePooling.

    Returns:
        ``(modelo de sentence-transformers, el mismo BERT en transformers, tokenizador)``.
        El segundo sirve para calcular la fórmula SPLADE a mano.
    """
    if not _TINY:
        d = TMP / "tiny"
        d.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(WORDS))
        # En transformers 5 el vocabulario se pasa con vocab= (con vocab_file= se ignora y
        # todo sale [UNK]).
        tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True)
        assert tok("the bitcoin zzz", add_special_tokens=False)["input_ids"] == [
            vocab.index("the"), vocab.index("bitcoin"), 1], "el tokenizador diminuto no reconoce su vocabulario"
        tok.save_pretrained(d)
        torch.manual_seed(0)
        cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=512)
        hf = BertForMaskedLM(cfg).eval()
        hf.save_pretrained(d)
        model = SparseEncoder(modules=[Transformer(str(d), transformer_task="fill-mask", max_seq_length=64),
                                       SpladePooling(pooling_strategy="max")], device="cpu")
        # Se registra para que SpladeIndex.encode_queries lo encuentre con get_model("tiny").
        splade._MODELS["tiny"] = model
        _TINY.update(model=model, hf=hf, tok=tok)
    return _TINY["model"], _TINY["hf"], _TINY["tok"]


def manual_splade(hf, tok, texts: list[str], max_len: int = 64) -> np.ndarray:
    """Fórmula SPLADE escrita a mano, como en el repo de IIMAS-RAG (``sparse.py``).

    Args:
        hf: modelo ``*ForMaskedLM``.
        tok: su tokenizador.
        texts: textos.
        max_len: truncado en tokens.

    Returns:
        Matriz densa (textos × V) con ``max_t log(1 + relu(logit_t)) · máscara_t``.
    """
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        logits = hf(**enc).logits
    w = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
    return w.max(dim=1).values.float().numpy()


def toy_index(name: str, dense_mat: np.ndarray, chunk_doc: list[int]) -> Path:
    """Guarda una matriz de juguete con ``save_matrix`` y devuelve su ruta.

    Args:
        name: nombre del archivo.
        dense_mat: matriz densa (trozos × V).
        chunk_doc: documento de cada trozo (contiguos).

    Returns:
        Ruta del ``.npz``.
    """
    path = TMP / f"{name}.npz"
    n_docs = max(chunk_doc) + 1
    splade.save_matrix(path, sp.csr_matrix(dense_mat.astype(np.float32)), chunk_doc, [f"d{i}" for i in range(n_docs)])
    return path


# ----------------------------------------------------------------------------- codificación

def test_encode_texts_matches_manual_formula():
    """CSR por bloques (bloques de 3, batch de 2) = fórmula a mano, en el orden original."""
    model, hf, tok = tiny()
    model.max_seq_length = 64
    got = splade.encode_texts(model, TEXTS, batch_size=2, block=3, kind="document").toarray()
    want = manual_splade(hf, tok, TEXTS)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(got, want, atol=1e-5), np.abs(got - want).max()
    # La versión de consulta usa la misma fórmula (estos modelos no tienen prompts).
    got_q = splade.encode_texts(model, TEXTS, batch_size=4, kind="query").toarray()
    assert np.allclose(got_q, want, atol=1e-5), np.abs(got_q - want).max()


def test_save_load_roundtrip_float16():
    """``save_matrix``/``load_matrix`` conservan la matriz (salvo el redondeo a float16) y los ids."""
    rng = np.random.default_rng(1)
    dense_mat = rng.random((7, 11)) * (rng.random((7, 11)) < 0.4) * 3
    path = toy_index("roundtrip", dense_mat, [0, 0, 1, 2, 2, 2, 3])
    mat, chunk_doc, doc_ids = splade.load_matrix(path)
    assert mat.dtype == np.float32
    assert np.allclose(mat.toarray(), dense_mat.astype(np.float16).astype(np.float32))
    assert list(chunk_doc) == [0, 0, 1, 2, 2, 2, 3] and doc_ids == ["d0", "d1", "d2", "d3"]
    assert not (TMP / "roundtrip.tmp.npz").exists()   # el temporal se renombró


# ----------------------------------------------------------------------------- búsqueda

def test_aggregations_match_brute_force():
    """``max``, ``first`` y ``maxpool`` dan lo mismo que el cálculo denso por fuerza bruta."""
    rng = np.random.default_rng(0)
    chunk_doc = [0, 0, 1, 2, 2, 2, 2, 3, 4, 4, 5, 5]
    starts = [0, 2, 3, 7, 8, 10]
    # Múltiplos de 1/8: exactos en float16 y sus productos suman sin redondeo en float32, así
    # que los scores del índice y los de la fuerza bruta son idénticos (también los empates).
    dmat = rng.integers(0, 9, size=(12, 30)) / 8 * (rng.random((12, 30)) < 0.3)
    qmat = rng.integers(0, 9, size=(5, 30)) / 8 * (rng.random((5, 30)) < 0.4)
    index = splade.SpladeIndex("toy", "toy", path=toy_index("agg", dmat, chunk_doc))
    qids = [f"q{i}" for i in range(5)]
    sims = qmat @ dmat.T                                                       # (consultas, trozos)
    pooled = np.stack([dmat[s:e].max(axis=0) for s, e in zip(starts, starts[1:] + [12])])
    brute = {
        "max": np.stack([sims[:, s:e].max(axis=1) for s, e in zip(starts, starts[1:] + [12])], axis=1),
        "first": sims[:, starts],
        "maxpool": qmat @ pooled.T,
    }
    for agg, scores in brute.items():
        run = index.search_vectors(qids, sp.csr_matrix(qmat.astype(np.float32)), k=6, agg=agg)
        for i, q in enumerate(qids):
            want = top_k(scores[i].astype(np.float32), index.doc_ids, 6)
            assert list(run[q]) == list(want), (agg, q, run[q], want)
            assert np.allclose(list(run[q].values()), list(want.values())), (agg, q)


def test_ties_follow_corpus_order():
    """Con scores iguales gana el documento que va antes en el corpus (como el oficial)."""
    dmat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    index = splade.SpladeIndex("toy", "toy", path=toy_index("ties", dmat, [0, 1, 2, 3]))
    run = index.search_vectors(["q"], sp.csr_matrix(np.array([[1.0, 0, 0]], dtype=np.float32)), k=4)
    assert list(run["q"]) == ["d1", "d3", "d0", "d2"], run


def test_prune_rows_keeps_top_n():
    """La poda deja los n pesos mayores de cada fila y no toca las filas cortas."""
    mat = sp.csr_matrix(np.array([[0.5, 0.1, 0.9, 0.3], [0.2, 0, 0, 0]], dtype=np.float32))
    pruned = splade.prune_rows(mat, 2).toarray()
    assert np.allclose(pruned, [[0.5, 0, 0.9, 0], [0.2, 0, 0, 0]]), pruned
    assert np.allclose(mat.toarray()[0], [0.5, 0.1, 0.9, 0.3])   # la original no cambia


def test_noexp_and_pruning_on_queries():
    """``-noexp`` deja solo los tokens de la consulta (con su peso) y ``-q2`` deja 2 términos."""
    _, hf, tok = tiny()   # tiny() también registra el modelo como "tiny" para encode_queries
    n_vocab = hf.config.vocab_size
    path = toy_index("tinyvocab", np.eye(4, n_vocab), [0, 1, 2, 3])
    text = ["the bitcoin halving price"]
    full = splade.SpladeIndex("tiny", "toy", path=path).encode_queries(text)
    noexp = splade.SpladeIndex("tiny", "toy", path=path, noexp=True).encode_queries(text)
    own = set(tok(text[0], add_special_tokens=False)["input_ids"])
    assert set(noexp.indices) <= own and len(noexp.indices) > 0, (set(noexp.indices), own)
    assert np.allclose(noexp.toarray()[0, noexp.indices], full.toarray()[0, noexp.indices])
    assert full.nnz > noexp.nnz   # el modelo aleatorio activa muchos términos ajenos a la consulta
    both = splade.SpladeIndex("tiny", "toy", path=path, noexp=True, q_max_dims=2).encode_queries(text)
    assert both.nnz == 2 and set(both.indices) <= set(noexp.indices)
    top2 = set(noexp.indices[np.argsort(-noexp.data)[:2]])
    assert set(both.indices) == top2


def test_end_to_end_tiny():
    """Codificar trozos → guardar → ``SpladeIndex.search_many`` = producto de la fórmula a mano."""
    model, hf, tok = tiny()
    model.max_seq_length = 64
    docs = TEXTS[:6]
    chunk_doc = [0, 1, 1, 2, 3, 3]                        # trozos contiguos por documento
    mat = splade.encode_texts(model, docs, batch_size=4, kind="document")
    path = TMP / "e2e.npz"
    splade.save_matrix(path, mat, chunk_doc, ["a", "b", "c", "d"])
    index = splade.SpladeIndex("tiny", "toy", path=path)
    queries = {"q1": "bitcoin halving price", "q2": "census records of the parish"}
    run = index.search_many(queries, k=4, agg="max")
    dmat = manual_splade(hf, tok, docs).astype(np.float16).astype(np.float32)   # lo guardado es float16
    qmat = manual_splade(hf, tok, list(queries.values()))
    for i, q in enumerate(queries):
        sims = qmat[i] @ dmat.T
        doc_scores = np.array([sims[0], sims[1:3].max(), sims[3], sims[4:6].max()])
        got = np.array([run[q][d] for d in ["a", "b", "c", "d"]])
        assert np.allclose(got, doc_scores, rtol=1e-4), (q, got, doc_scores)


# ----------------------------------------------------------------------------- nombres

def test_parse_retriever():
    """Gramática ``<modelo>[-q<N>][-noexp][@agg]``; los nombres no SPLADE devuelven None."""
    p = splade.parse_retriever
    assert p("splade-ed") == {"key": "splade-ed", "q_max_dims": None, "noexp": False, "agg": "max"}
    assert p("splade-ed@first")["agg"] == "first"
    assert p("splade-v3-db-q64-noexp@maxpool") == {"key": "splade-v3-db", "q_max_dims": 64,
                                                   "noexp": True, "agg": "maxpool"}
    assert p("splade-v3-q128") == {"key": "splade-v3", "q_max_dims": 128, "noexp": False, "agg": "max"}
    assert p("bm25") is None and p("bge-base@first") is None and p("splade-xx") is None
    for bad in ("splade-ed@sum", "splade-ed-q0"):
        try:
            p(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} debería dar ValueError")


# ----------------------------------------------------------------------------- modelo real (opcional)

def test_real_splade_ed():
    """Con ``splade-ed`` real: ST = fórmula a mano, y los trozos son los mismos que con bge-base.

    Solo corre con ``RETECO_SPLADE_REAL=1`` (descarga el modelo, ~440 MB). En CPU y fp32 para
    poder comparar con tolerancia baja.
    """
    if os.environ.get("RETECO_SPLADE_REAL") != "1":
        raise Skip("definir RETECO_SPLADE_REAL=1 para correrla")
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    name = splade.MODELS["splade-ed"]
    model = SparseEncoder(name, device="cpu")
    model.max_seq_length = 64
    texts = ["When was the Treaty of Paris signed, and did it end the war in 1783?",
             "Bitcoin's 2020 halving reduced the block reward to 6.25 BTC.",
             "Parish records and the 1841 census of England list the family in Yorkshire."]
    got = splade.encode_texts(model, texts, batch_size=2, kind="document").toarray()
    want = manual_splade(AutoModelForMaskedLM.from_pretrained(name).eval(), AutoTokenizer.from_pretrained(name), texts)
    assert np.allclose(got, want, atol=1e-4), np.abs(got - want).max()
    nnz = (got > 0).sum(axis=1)
    assert ((nnz > 20) & (nnz < 2000)).all(), nnz   # disperso de verdad: decenas o cientos de términos
    # Mismo vocabulario y mismos trozos que bge-base (PLAN_SPLADE.md §3.2).
    bge = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    assert model.tokenizer.get_vocab() == bge.get_vocab()
    long_texts = [" ".join(texts * 40), "Café déjà vu — naïve résumé 1834–1840 https://x.org/a?b=1", ""]
    assert chunk_texts(model.tokenizer, long_texts) == chunk_texts(bge, long_texts)
    print(f"      términos activos por texto: {nnz.tolist()}")


if __name__ == "__main__":
    # Ejecutor mínimo sin pytest: corre todas las funciones test_* en orden alfabético.
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok      {name}")
        except Skip as exc:
            skipped += 1
            print(f"omitida {name}: {exc}")
        except Exception as exc:   # noqa: BLE001 — se informa de cualquier fallo y se sigue
            failed += 1
            print(f"FALLA   {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} pruebas correctas, {skipped} omitida(s)")
    sys.exit(1 if failed else 0)
