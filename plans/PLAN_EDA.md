# Plan de exploración de datos — Track 1 (TEMPO), sub-tracks 1a y 1b

Objetivo: entender el corpus, las consultas y los juicios de relevancia de la Tarea 1 de RETECO, para tomar decisiones informadas sobre el pipeline (retriever, reranker, uso del razonamiento temporal).

## 0. Contexto de la tarea

| | 1a · Temporal retrieval | 1b · Step-wise retrieval |
|---|---|---|
| Qué se recupera | documentos para la **consulta completa** | documentos para **cada paso** descompuesto de la consulta |
| Entrada | `examples_{train,dev}.jsonl` | `steps_{train,dev}.jsonl` |
| Juicios | `qrels_{train,dev}.txt` (qid = `id`) | `qrels_steps_{train,dev}.txt` (qid = `step_id`) |
| Métrica | nDCG@10 (`pytrec_eval`), macro-promedio por dominio | igual |
| Baseline BM25 (dev) | 0.1147 | 0.1177 |

Datos generales (README, v1.1): 13 dominios, 1,167,159 documentos, 1,730 consultas (1,211 train / 519 dev), 3,976 pasos (2,762 / 1,214). El split es 70/30 por consulta; el corpus de cada dominio es único y compartido por train y dev.

Archivos por dominio (`data/RETECO-SemEval2027/track1_tempo/<dominio>/`):

- `documents.jsonl`: `{id, content}`
- `examples_*.jsonl`: `{id, query, gold_ids, gold_answers}` (1a)
- `steps_*.jsonl`: `{id, query, steps:[{step_id, step, step_instruction, gold_ids}]}` (1b)
- `guidance_*.jsonl`: metadatos temporales de la consulta (`query_guidance`) y anotaciones de los pasajes gold (`gold_passage_annotations`)
- `duplicate_map.json`: ID duplicado eliminado → ID conservado

**Regla de trabajo:** toda la exploración y el ajuste se hacen sobre **train**. Dev se usa solo para validar al final, no para tomar decisiones de diseño.

---

## 1. Inventario y estructura (sanity checks)

- [ ] Conteos por dominio: documentos, consultas train/dev, pasos train/dev, líneas de qrels.
- [ ] Validar esquemas: campos presentes, tipos y nulos en cada archivo.
- [ ] Integridad de IDs:
  - todo `gold_id` de examples/steps existe en `documents.jsonl`;
  - qrels ↔ `gold_ids` coinciden (mismos pares);
  - cada `step_id` tiene la forma `<query_id>_step<n>`;
  - no hay IDs de `duplicate_map.json` presentes en qrels (deben estar ya mapeados).
- [ ] Comparar los IDs de `split_manifest.json` con los archivos y confirmar que no hay solapamiento train/dev.
- [ ] Relación 1a ↔ 1b: ¿todas las consultas de `examples` están en `steps`? ¿Cuántas tienen pasos con gold y cuántas no? (Ya se ve un caso de `law` con solo `step3`, así que hay pasos faltantes o filtrados.)

**Salida:** tabla resumen por dominio y lista de anomalías.

Dominios por tamaño de corpus (docs / consultas train+dev):

| Dominio | Docs | Consultas |
|---|---:|---:|
| history | 200,252 | 801 |
| hsm | 169,552 | 150 |
| travel | 158,771 | 100 |
| genealogy | 136,101 | 115 |
| politics | 135,207 | 150 |
| bitcoin | 76,706 | 100 |
| economics | 74,692 | 83 |
| monero | 53,955 | 65 |
| cardano | 47,509 | 51 |
| workplace | 43,584 | 36 |
| law | 36,110 | 35 |
| quant | 25,040 | 34 |
| iota | 9,680 | 10 |

`history` concentra casi la mitad de las consultas, pero la métrica se promedia **por dominio**: los dominios pequeños (`iota`, `quant`, `law`) pesan lo mismo y sus estimaciones serán ruidosas.

## 2. Corpus (`documents.jsonl`)

- [ ] Longitud de documentos (caracteres y tokens) por dominio: distribución, percentiles y outliers. Esto define si hay que fragmentar (chunking) y el `max_seq_length` de los encoders.
- [ ] Documentos vacíos o casi vacíos, boilerplate (licencias, disclaimers, menús) y casi-duplicados que sobrevivieron a la deduplicación exacta (por ejemplo, con MinHash).
- [ ] Formato: ¿HTML, markdown o texto plano? Idioma y caracteres raros.
- [ ] Estructura de los IDs (`<dominio>/<hash>_<n>.txt` vs `<dominio>/<qid>_<hash>_<n>.txt`): qué significan el prefijo y el número. Comprobar si los gold comparten un patrón de ID; si lo comparten, **no debe explotarse** (sería leakage que no se generaliza al test).
- [ ] Densidad temporal: proporción de documentos con fechas, años o expresiones temporales (regex primero y, si hace falta, un tagger tipo SUTime/HeidelTime o spaCy). Comparar gold con no-gold.
- [ ] Gold vs. no-gold: longitud, densidad temporal, vocabulario. ¿Los gold son distinguibles por rasgos superficiales?

## 3. Consultas 1a (`examples_*.jsonl`)

- [ ] Las consultas vienen en HTML (título + cuerpo estilo StackExchange): decidir la limpieza (quitar tags y separar título y cuerpo). Medir la longitud resultante.
- [ ] Número de gold por consulta (distribución): afecta al techo de nDCG@10 y a la elección de k.
- [ ] `gold_answers`: longitud y utilidad. Pueden servir para entrenar o generar consultas sintéticas (HyDE), pero **no estarán en test**.
- [ ] Expresiones temporales explícitas en la consulta: años, "before/after", "recently", rangos.
- [ ] Solapamiento léxico consulta ↔ gold (por ejemplo, Jaccard o BM25 score del gold). Esto explica por qué BM25 rinde poco y cuánta semántica hace falta.

## 4. Pasos 1b (`steps_*.jsonl`)

- [ ] Número de pasos por consulta y número de gold por paso.
- [ ] Longitud y estilo de `step_instruction`: ¿son instrucciones ("Consult…", "Identify…") más que preguntas? Esto puede requerir reformularlas para el retriever.
- [ ] ¿La instrucción del paso basta sola, o necesita la consulta original como contexto? Comparar con BM25 rápido usando paso solo, paso + consulta y consulta sola.
- [ ] Solapamiento de gold entre pasos de la misma consulta y entre los gold de 1b y los de 1a de esa consulta.
- [ ] Pasos "huecos": relacionar la numeración (`step` 1..n) con los pasos presentes para entender qué se filtró.

## 5. Guidance temporal (`guidance_*.jsonl`)

Es información extra muy rica. Hay que confirmar si estará en test; probablemente **no**, así que se trata como señal de entrenamiento y análisis, no como input de inferencia.

- [ ] Distribución de `temporal_intent`, `temporal_reasoning_class_primary/secondary`, `expected_granularity` e `is_temporal_query`, por dominio.
- [ ] `retrieval_plan` vs `steps`: ¿los pasos de 1b salen de aquí? ¿Cuántas acciones tienen `gold_ids` vacíos?
- [ ] `gold_passage_annotations`: `tense_guess`, `time_scope_guess`, `time_mentions` y `confidence`. ¿Qué tan bien coinciden las anclas temporales de la consulta con las de los pasajes gold?
- [ ] Rendimiento del baseline segmentado por clase temporal, para identificar los tipos de consulta más difíciles.

## 6. Baselines y diagnóstico de errores (train)

- [ ] Correr el baseline oficial del starter kit (BM25 Lucene) sobre train y reproducir ~0.1075 (1a) y ~0.1024 (1b).
- [ ] Probar un baseline denso rápido (por ejemplo, `bge`/`e5` con `sentence-transformers` + FAISS) en 1–2 dominios pequeños (`iota`, `quant`, `law`).
- [ ] Recall@100 y @1000 de cada baseline. Define si el cuello de botella está en la recuperación (primera etapa) o en el ranking (reranker).
- [ ] Revisar a mano ~20 consultas fallidas: ¿el gold no aparece por desajuste léxico, por la temporalidad o por consultas largas y ruidosas?
- [ ] nDCG@10 por dominio y varianza, especialmente en los dominios pequeños.

## 7. Entregables

- `notebooks/01_inventario.ipynb`: secciones 1–2
- `notebooks/02_consultas_pasos.ipynb`: secciones 3–4
- `notebooks/03_guidance_temporal.ipynb`: sección 5
- `notebooks/04_baselines.ipynb`: sección 6
- `src/data.py`: cargadores reutilizables (documentos, consultas, pasos, qrels, guidance) para no repetir código
- Resumen de hallazgos (al final de este archivo o en `HALLAZGOS.md`) con decisiones para el pipeline

## Preguntas abiertas (confirmar en la web de la tarea o con los organizadores)

1. ¿Qué campos estarán disponibles en el **test** para 1a y 1b? ¿Habrá `guidance`? ¿En 1b se dan los `step_instruction`?
2. ¿Se puede usar el corpus de un dominio para otro o datos externos? (Cada dominio es "self-contained".)
3. Número máximo de documentos por ranking en la submission (¿top-100? ¿top-1000?).

## Notas prácticas

- `history` y `hsm` (~200k y ~170k docs) son los más caros de indexar con modelos densos. Conviene prototipar en dominios pequeños y escalar después.
- Cargar `documents.jsonl` en streaming o con `pandas.read_json(..., lines=True)` por dominio, no todos a la vez (3 GB en total).
- Instalar `pytrec_eval` para la métrica oficial: `uv add pytrec-eval-terrier`.
