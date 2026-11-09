# Bibliotecas del proyecto RAG (SemEval 2027 – RETECO, Tarea 1)

Resumen de las dependencias instaladas, para qué sirven en el proyecto y su documentación oficial.

## Gestión de entorno y datos

| Biblioteca | Para qué sirve | Documentación |
|---|---|---|
| **uv** | Gestor de paquetes y entornos virtuales de Python (reemplaza a pip + venv). Maneja `pyproject.toml` y el lockfile. | https://docs.astral.sh/uv/ |
| **huggingface_hub** | Cliente para Hugging Face Hub. Aporta el comando `hf` usado para descargar el dataset RETECO y gestionar el login. | https://huggingface.co/docs/huggingface_hub |
| **datasets** | Carga y manipula datasets (formatos parquet/json/csv, mapeo, filtrado). Permite `load_dataset("DataScience-UIBK/RETECO-SemEval2027")`. | https://huggingface.co/docs/datasets |
| **pandas** | Análisis y manipulación de datos tabulares (DataFrames). Útil para explorar el dataset y armar resultados. | https://pandas.pydata.org/docs/ |
| **numpy** | Arreglos numéricos y álgebra. Base de embeddings y métricas. | https://numpy.org/doc/ |
| **pyarrow** | Lectura/escritura de parquet y memoria columnar. Lo usa `datasets` por debajo. | https://arrow.apache.org/docs/python/ |
| **tqdm** | Barras de progreso para bucles largos (indexado, embeddings, inferencia). | https://tqdm.github.io/ |
| **scikit-learn** | Métricas, utilidades de evaluación, TF-IDF y preprocesamiento clásico. | https://scikit-learn.org/stable/ |

## Modelos y deep learning

| Biblioteca | Para qué sirve | Documentación |
|---|---|---|
| **torch (PyTorch)** | Framework de tensores y redes neuronales. Motor de los modelos locales; en Mac usa la GPU vía MPS. | https://pytorch.org/docs/stable/ |
| **transformers** | Modelos preentrenados (LLMs, encoders, rerankers) con tokenizadores y utilidades de generación. Sirve para el generador y el reranker. | https://huggingface.co/docs/transformers |
| **accelerate** | Simplifica la ejecución de modelos en distintos dispositivos y la carga de modelos grandes. | https://huggingface.co/docs/accelerate |
| **sentence-transformers** | Genera embeddings de texto (bi-encoders) y cross-encoders para reranking. Es la base del retriever denso. | https://www.sbert.net/ |

## Recuperación (retrieval)

| Biblioteca | Para qué sirve | Documentación |
|---|---|---|
| **faiss-cpu** | Búsqueda de similitud eficiente sobre vectores (índices flat, IVF, HNSW). Almacena y consulta los embeddings. | https://faiss.ai/ · https://github.com/facebookresearch/faiss/wiki |
| **rank-bm25** | Implementación ligera de BM25 en Python puro. Línea base léxica rápida de armar. | https://github.com/dorianbrown/rank_bm25 |
| **pyserini** | Toolkit de recuperación sobre Lucene/Anserini: BM25 a escala, retrieval denso e híbrido, evaluación estilo TREC. Requiere JDK 21. | https://github.com/castorini/pyserini · https://castorini.github.io/pyserini/ |
| **nltk** | Procesamiento de lenguaje natural clásico: tokenización, stopwords, stemming. Útil para preprocesar texto para BM25. | https://www.nltk.org/ |

## Generación vía API y utilidades

| Biblioteca | Para qué sirve | Documentación |
|---|---|---|
| **anthropic** | SDK oficial de la API de Claude. Opcional: generador del RAG o juez de evaluación. | https://docs.claude.com/ · https://github.com/anthropics/anthropic-sdk-python |
| **python-dotenv** | Carga variables de entorno desde un archivo `.env` (por ejemplo, `ANTHROPIC_API_KEY`) sin escribirlas en el código. | https://saurabh-kumar.com/python-dotenv/ |

## Desarrollo

| Biblioteca | Para qué sirve | Documentación |
|---|---|---|
| **jupyter** | Notebooks interactivos para explorar datos y prototipar. | https://docs.jupyter.org/ |
| **ipykernel** | Kernel de Python para Jupyter; permite elegir el entorno `.venv` en los notebooks. | https://ipython.readthedocs.io/en/stable/ |
| **ruff** | Linter y formateador de código muy rápido. | https://docs.astral.sh/ruff/ |

## Requisito externo

| Herramienta | Para qué sirve | Documentación |
|---|---|---|
| **OpenJDK 21** | Entorno de ejecución de Java que necesita Pyserini (Lucene/Anserini). | https://openjdk.org/projects/jdk/21/ |

## Flujo típico en el RAG

1. **Datos:** `huggingface_hub` / `datasets` → `pandas`
2. **Indexado:** `sentence-transformers` + `faiss-cpu` (denso) y `pyserini` / `rank-bm25` (léxico)
3. **Recuperación:** búsqueda densa, léxica o híbrida
4. **Reranking:** cross-encoder de `sentence-transformers` o modelo de `transformers`
5. **Generación:** LLM local con `transformers`, o `anthropic` vía API
6. **Evaluación:** `scikit-learn` / `pandas`, y las métricas oficiales de la tarea

> Nota: los enlaces son las páginas oficiales conocidas de cada proyecto; verifica que sigan vigentes, ya que las URLs de documentación pueden cambiar.
