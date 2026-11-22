# RETECO · Recuperación de documentos con razonamiento temporal — Proyecto final de PLN

Proyecto final de la materia de **Procesamiento del Lenguaje Natural**. Trabaja sobre la Tarea 1 de RETECO (SemEval-2027), el *track* **TEMPO**: dada una pregunta de StackExchange que exige razonar sobre el tiempo ("¿qué se sabía de X antes de 1600?", "¿cuándo cambió Y?"), hay que recuperar los documentos que la responden dentro del corpus de su dominio.

- **13 dominios**: `bitcoin`, `cardano`, `economics`, `genealogy`, `history`, `hsm`, `iota`, `law`, `monero`, `politics`, `quant`, `travel` y `workplace` (1.17 M documentos en total).
- **Subtarea 1a**: se busca con la consulta completa (1,211 consultas en train).
- **Subtarea 1b**: cada consulta viene descompuesta en pasos ("Search for primary sources on…") y se busca con cada paso (2,762 pasos en train).
- **Métrica**: nDCG@10, como media de las medias por dominio (*macro*). También R@100 y R@1000.

**Restricción del proyecto:** solo se usan técnicas vistas en clase. BM25 se incluye porque es el baseline oficial de la tarea y es una variante de TF-IDF (§1).

**Reglas de trabajo:** todo se decide en **train**; dev se usó una sola vez, al final, con la configuración ya fijada (§2). Nunca se usan en la búsqueda el formato de los IDs de los documentos, las respuestas aceptadas (`gold_answers`) ni los metadatos `guidance`, que no estarán en test. Esos campos solo se usan para medir techos (oráculos) y para analizar resultados.

---

## 1. Técnicas de la materia y dónde se usan

| Técnica | Dónde se usa |
|---|---|
| **Expresiones regulares** | limpieza del HTML de las consultas; `strip_step` quita el verbo imperativo y las palabras meta de los pasos de 1b ([`queries.py`](src/reteco/queries.py)); extractor de expresiones temporales (años, meses, décadas, siglos, rangos, a. C.) y de su papel (cita, metadato) en [`temporal.py`](src/reteco/temporal.py); K4 |
| **TF-IDF** | BM25 es TF-IDF con la frecuencia del término saturada y normalizada por la longitud del documento ([`BASELINE.md`](BASELINE.md)); prueba de clasificador TF-IDF (1-2 gramas) + regresión logística para el tipo de razonamiento ([`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md) §0.1) |
| **Embeddings** | recuperador denso con `bge-base-en-v1.5`: consultas y trozos de documento como vectores y búsqueda por producto escalar ([`dense.py`](src/reteco/dense.py)) |
| **N-gramas** | unigramas y bigramas del perfil temporal implícito (TT0, [`RESULTADOS_PERFIL_TEMPORAL.md`](RESULTADOS_PERFIL_TEMPORAL.md)); 1-2 gramas del clasificador TF-IDF |
| **Cadenas de Markov** | en TT0, `P(periodo | n-grama)` es la matriz de transición de un paso de una cadena n-grama → periodo, estimada contando co-ocurrencias en oraciones fechadas |
| **Jaccard** | EDA: casi-duplicados del corpus con MinHash + LSH (estimación de Jaccard) y solapamiento de vocabulario entre consulta y gold ([`notebooks/EDA_EXPLICADO.md`](notebooks/EDA_EXPLICADO.md)) |

Además se usan técnicas estándar de evaluación y análisis: fusión de rankings con *Reciprocal Rank Fusion* (RRF), test pareado de aleatorización, validación cruzada y log-odds con prior de Dirichlet para comparar vocabularios.

## 2. Resultados principales (macro nDCG@10)

Todo se eligió en train. Las columnas de dev son una única corrida final, sin ajustar nada después.

| Sistema | 1a train | 1a dev | 1b train | 1b dev |
|---|---:|---:|---:|---:|
| Baseline oficial (BM25 con la consulta cruda) | 0.1075 | 0.1147 | 0.1024 | 0.1177 |
| BM25 con la mejor variante de consulta (sin modelos) | 0.1528 | 0.1716 | 0.1660 | 0.1838 |
| Denso `bge-base` | 0.2277 | 0.2226 | 0.2449 | 0.2610 |
| **Híbrido BM25 + denso** (RRF, BM25 con peso 0.5) | **0.2367** | **0.2519** | **0.2610** | **0.2971** |

- **Preparar la consulta ya rinde mucho con BM25**: buscar también con el título (1a) o quitar al paso el verbo imperativo y las palabras meta (1b) sube +0.045 / +0.064 sin ningún modelo.
- **Los embeddings duplican el baseline**, y el híbrido con BM25 a medio peso es la mejor primera etapa (2.2–2.5 veces el baseline).
- **Dev confirma la progresión.** Cada paso mejora al anterior de forma significativa, y el híbrido supera al denso solo en las dos subtareas (+0.029 y +0.036, p < 0.001). En train esa mejora solo era significativa en 1b.
- **Usar el tiempo o el tipo de razonamiento de forma explícita no mejora el híbrido**: ni clasificar el tipo de razonamiento para elegir la estrategia (O4), ni un vocabulario de documento por clase (O5), ni añadir la década y el siglo de las fechas de la consulta (K4), ni el perfil temporal implícito con n-gramas (TT0). El porqué está en cada sección.

Detalle completo: [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md).

## 3. Documentación

| Documento | Contenido |
|---|---|
| [`notebooks/EDA_EXPLICADO.md`](notebooks/EDA_EXPLICADO.md) | análisis exploratorio de los datos, explicado (notebook: [`eda_track1_tempo.ipynb`](notebooks/eda_track1_tempo.ipynb)) |
| [`BASELINE.md`](BASELINE.md) | cómo funciona el baseline oficial y qué enfoques prueba el paper de TEMPO (notebook: [`baseline_track1_tempo.ipynb`](notebooks/baseline_track1_tempo.ipynb)) |
| [`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md) | plan de experimentos: tipo de razonamiento, keywords y variantes de consulta |
| [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) | resultados: variantes de consulta con BM25, oráculos, clasificación del razonamiento (O4, O5), denso, híbrido, K4 y comprobación en dev |
| [`reporte/`](reporte/) | reporte tipo paper: [`reporte.tex`](reporte/reporte.tex), [`reporte.pdf`](reporte/reporte.pdf) y [`reporte.docx`](reporte/reporte.docx), con su bibliografía y [`compilar.sh`](reporte/compilar.sh) |
| [`RESULTADOS_PERFIL_TEMPORAL.md`](RESULTADOS_PERFIL_TEMPORAL.md) | perfil temporal implícito con n-gramas y cadenas de Markov (TT0) |
| [`SETUP.md`](SETUP.md) | instalación del entorno y descarga del dataset |
| [`plans/PLAN_EDA.md`](plans/PLAN_EDA.md) | plan del EDA |

## 4. Estructura del código

```
src/reteco/
  data.py       rutas, cargadores (consultas, pasos, qrels, guidance), limpieza HTML, etiquetas
  bm25.py       réplica exacta del BM25 del baseline oficial
  dense.py      embeddings bge (trozos de 254 tokens) y búsqueda por producto escalar
  fusion.py     top-k con el desempate del oficial y RRF
  queries.py    variantes de consulta de 1a y 1b (incluidas las de oráculo y las de K4)
  temporal.py   expresiones temporales con regex y formas gruesas (K4)
  evaluate.py   nDCG y recall con pytrec_eval, macro por dominio, test pareado, registro
scripts/
  run_experiment.py        corre variantes de consulta con BM25 o con el denso y las evalúa
  encode_corpus.py         codifica los corpus con el encoder denso
  fuse_runs.py             híbrido: fusiona runs ya calculados con RRF
  oracle_routing.py        O4: ¿sirve clasificar la consulta para elegir la estrategia?
  oracle_class_terms.py    O5: ¿hay keywords de documento propias de cada clase?
  k4_diag.py               K4: desglose con fecha / sin fecha y por dominio
  temporal_profile_diag.py TT0: léxico n-grama → periodo y perfiles temporales
tests/
  test_extraction.py       pruebas del extractor temporal y de K4
```

`bm25` arranca la JVM de Pyserini y `dense` usa torch: se ejecutan siempre en procesos distintos, porque juntos hacían caer el kernel en el EDA.

## 5. Reproducción

Instalación y datos: [`SETUP.md`](SETUP.md). Después, por ejemplo:

```bash
.venv/bin/python tests/test_extraction.py                                    # pruebas
.venv/bin/python scripts/run_experiment.py --subtrack 1a --retriever bm25 --variants all
.venv/bin/python scripts/encode_corpus.py --model bge-base                   # ~4.6 h en un M2 Pro
.venv/bin/python scripts/run_experiment.py --subtrack 1a --retriever bge-base --variants all
.venv/bin/python scripts/fuse_runs.py --subtrack 1a --weights 0.5 1 \
    --runs "bm25/rrf(limpia,titulo)" "bge-base/rrf(limpia,titulo)"         # híbrido
```

Para la comprobación en dev se repiten los mismos comandos con `--split dev` y solo con las variantes finales (lista completa en [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) §2.12).

Los runs se guardan en `outputs/exp/runs/` y cada experimento añade una fila a `outputs/exp/resultados.csv` (ninguno de los dos se sube a git).

## 6. Referencias

- A. Abdallah, M. Ali, M. Abdul-Mageed y A. Jatowt. *TEMPO: A Realistic Multi-Domain Benchmark for Temporal Reasoning-Intensive Retrieval.* arXiv:2601.09523. [`papers/2601.09523v1.pdf`](papers/2601.09523v1.pdf). El benchmark de la tarea.
- A. Jatowt, C.-M. Au Yeung y K. Tanaka. 2013. *Estimating Document Focus Time.* CIKM '13, pp. 2273–2278. [`papers/jatowt2013_focus_time.pdf`](papers/jatowt2013_focus_time.pdf). Base del perfil temporal implícito (TT0).
- J. Han, A. Cheung, Y. Wei, Z. Yu, X. Wang, B. Zhu y Y. Yang. 2025. *RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval for Evolving Knowledge* (TG-RAG). arXiv:2510.13590. [`papers/2510.13590v1.pdf`](papers/2510.13590v1.pdf). Trabajo relacionado: representa el corpus como un grafo de hechos con su fecha, organizado en una jerarquía de tiempo, y lo recorre con PageRank. De ahí salió la idea de combinar n-gramas con un paso por una cadena de Markov en TT0.
