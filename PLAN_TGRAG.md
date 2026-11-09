# Plan — Temporal GraphRAG (TG-RAG) para el Track 1 (TEMPO)

Paper: Han et al., *RAG Meets Temporal Graphs: Time-Sensitive Modeling and Retrieval for Evolving Knowledge* ([`papers/2510.13590v1.pdf`](papers/2510.13590v1.pdf), arXiv 2510.13590, oct. 2025). Código oficial: `github.com/hanjiale/Temporal-GraphRAG` (no se usa: se reimplementa lo necesario sobre `src/reteco/`).

Las reglas no cambian (ver [`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md) §1):
- todo se ajusta en train y dev solo sirve para confirmar al final;
- en inferencia no se usan el formato de los IDs, `gold_answers` ni `guidance`;
- `guidance` solo aparece en las variantes marcadas como **oráculo**.

---

## 1. Qué propone el paper

TG-RAG representa el corpus como un **grafo temporal de dos niveles**:

1. **Indexación**
   - Un LLM extrae de cada chunk **cuádruplas** `(entidad₁, entidad₂, relación, τ)`, con τ un timestamp normalizado.
   - Nivel inferior: un grafo de conocimiento cuyas aristas llevan timestamp. El mismo par de entidades en fechas distintas da **aristas paralelas distintas**.
   - Nivel superior: una **jerarquía de tiempo** (año → trimestre → mes → día). Cada nodo de tiempo se enlaza con las aristas activas en ese momento.
   - Se calculan embeddings de las descripciones de las relaciones.
   - Un LLM escribe un **informe por nodo de tiempo**, de abajo arriba.
2. **Recuperación local** (Algoritmo 1)
   1. Un LLM extrae del texto de la consulta su alcance temporal `T_q` (`at` / `before` / `after` / `between`).
   2. Se toman las top-K aristas por coseno `γ_ε = cos(e_q, e_ε)` (K = 20). Forman el subgrafo `G_q`.
   3. Semillas: las entidades de las aristas de `G_q` cuyo τ ∈ `T_q`.
   4. Se ejecuta **PPR** sobre `G_q` con esas semillas, lo que da `s(v)`.
   5. Score de cada arista: `s(ε) = 1[τ ∈ T_q] · (s(v₁) + s(v₂))`.
   6. Score de cada chunk: `s(c) = Π_{ε∈E(c)} (1 + γ_ε) · Σ_{ε∈E(c)} s(ε)`.
   7. Los chunks se ordenan por `s(c)`.
3. **Recuperación global:** informes de los nodos de `T_q`, puntos atómicos extraídos con un LLM y una síntesis final (para preguntas abstractas).
4. **Actualización incremental:** solo se extraen documentos nuevos y solo se regeneran los informes de los nodos de tiempo afectados.

**Resultados relevantes para nosotros** (ECT-QA, tabla VII, ablaciones):

| Variante | Correct |
|---|---:|
| Completo | 0.599 |
| Sin PPR | 0.580 |
| **Sin filtro temporal** | **0.382** |
| Sin filtro ni PPR | 0.482 |
| Grafo estático | 0.381 |

- El **filtro temporal en la consulta** es la pieza que más aporta.
- El PPR sin filtro temporal *empeora*: propaga ruido.
- En QA genérico (NQ, HotpotQA), el filtro se desactiva cuando la consulta no tiene un alcance temporal claro.

> Nota: la ecuación (3) del paper escribe `Σ_{ε∈E_q}`, pero esa suma sería la misma para todos los chunks. Se interpreta como `Σ_{ε∈E(c)}`, coherente con la definición de `E(c)` justo antes.

## 2. ¿Encaja con TEMPO? Diferencias que obligan a adaptar

| Aspecto | ECT-QA (paper) | TEMPO Track 1 | Consecuencia |
|---|---|---|---|
| Tarea | QA: generar la respuesta | **ranking de documentos** (nDCG@10) | sobra la generación; el score de chunk se convierte en score de documento |
| Corpus | 480 transcripciones, 1.6 M tokens | **1.17 M documentos**, ~2.5 GB (`history`: 200 k docs) | extraer con LLM todo el corpus es inviable en local (el paper gastó 6.3 M tokens de prompt con 1.6 M de corpus; aquí serían cientos de millones) → **TG-RAG como reranker sobre un pool de candidatos** |
| Tiempo | trimestres fiscales explícitos | siglos, décadas, años, "prior to 1600", BCE; **el 55 % de los pares (consulta, gold) no comparte año** (EDA 6.3) | jerarquía milenio → siglo → década → año → (trimestre) → mes → día; **filtro duro arriesgado** → probar también un emparejamiento suave |
| Consultas | cortas y con fecha | largas (mediana 127 palabras); ~80 % con alguna marca temporal, pero sobre todo relativa ("before", "since") | extracción de `T_q` con reglas y con LLM; *fallback* sin filtro, como hace el paper en QA genérico |
| Documentos | 1,200 tokens por chunk | mediana ~200 tokens | chunk ≈ documento (los largos se parten en trozos de ~800 palabras) |

**Fuera de alcance**, y por qué:
- **Recuperación global e informes por nodo de tiempo.** Resumen muchos documentos y no se pueden puntuar como un `doc_id`. Lo más parecido sería usarlos como expansión tipo HyDE, pero O1 mostró que el texto largo hunde a BM25. Queda como extensión opcional (§7).
- **Actualización incremental.** El corpus de TEMPO es estático. Aun así, la extracción es incremental por documento: si el pool crece, solo se procesan los documentos nuevos.

### Hipótesis

H1. En las consultas con anclas temporales explícitas, la combinación **"semántica de la relación + coincidencia de periodo + propagación por entidades"** reordena mejor el top del pool que el retriever de primera etapa. Se espera ganancia sobre todo en EAL y TPC, y en `genealogy`, `politics` y `history`.

H2. Igual que en el paper, **el componente temporal es el que más aporta**. Si sin él no hay mejora, el grafo solo es un reranker semántico más (y para eso habrá un cross-encoder).

H3. La extracción con LLM es mejor que la de reglas, pero la de reglas captura la mayor parte de la señal temporal. La señal la ponen los timestamps, que se extraen bien con regex. Las entidades capitalizadas son una aproximación razonable.

## 3. Diseño adaptado

```
primera etapa (runs ya calculados)
  └─ pool por consulta: unión del top-N de 1–2 runs (N = 100)
       └─ extracción de hechos temporales en los documentos del pool   [reglas | LLM]
            └─ grafo bi-nivel por dominio (entidades, aristas con τ, índice nodo-de-tiempo → aristas)
                 └─ embeddings de las descripciones de arista (bge-small por defecto)
consulta
  ├─ T_q: alcance temporal                                                [reglas | LLM | oráculo]
  ├─ e_q: embedding de la consulta (variante `limpia` o `titulo+paso_limpio`)
  └─ Algoritmo 1 adaptado → score TG por documento
       └─ fusión con la primera etapa (RRF ponderado) → run top-1000 → pytrec_eval
```

### 3.1 Extracción de hechos (`reteco.tgraph`)

**Con reglas** (TG-lite, sin LLM; se aplica a todo el pool):
- Oraciones por regex.
- Expresiones temporales con `reteco.temporal`: fechas, mes-año, años, rangos, décadas, siglos (también "early/mid/late"), BCE/AD y trimestres.
- Entidades: secuencias capitalizadas (permitiendo "of", "the", "de"…) y acrónimos, quitando meses, días y palabras de inicio de oración.
- Cada oración con ≥ 1 entidad genera aristas en cadena `(eᵢ, eᵢ₊₁)` (o `(e, e)` si solo hay una).
- τ: cada expresión temporal de la oración; `None` si no tiene ninguna. Estos hechos sin tiempo se conservan, como en la variante genérica del paper.
- La descripción de la arista es la propia oración.

**Con LLM** (fiel al paper; solo en un subconjunto):
- Prompt propio, porque el paper no publica el de extracción en el PDF.
- Pide hechos JSON `{subject, object, relation, time, description}` con `time` en un formato normalizado (`1834`, `1834-03`, `1830s`, `C19`, `-300`, `1914/1918`) o `null`.
- Cliente con caché en disco (`reteco.llm`, backend `mlx` o compatible con OpenAI).

Los hechos se guardan por documento en `outputs/exp/tgrag/facts/<extractor>/<dominio>.jsonl`.

### 3.2 Tiempo (`reteco.temporal`)

- `TimeSpan(start, end, gran)`: intervalo de años [inicio, fin) con su granularidad.
- Nodos de la jerarquía: `century:1800`, `decade:1830`, `year:1834`, `month:1834-03`…
- `T_q` es una lista de `Scope(logic, span)`. La lógica se detecta por las palabras previas: before / prior to / until / pre- → `before`; after / since / post- → `after`; between / from…to → rango.
- Emparejamiento τ ↔ `T_q` (parámetro `match`):
  - `contained`: τ cae dentro del alcance (descendiente o igual en la jerarquía). Equivale al paper.
  - `overlap`: τ solapa el alcance (incluye ancestros: "19th century" para una consulta sobre 1834).
  - `soft`: fracción de τ dentro del alcance; si no hay solape, decae como `exp(−distancia / escala)`.
- `T_q` **oráculo**: el mismo parser aplicado a `key_time_anchors` y `query_temporal_signals` de guidance. Es el techo de la identificación temporal.

### 3.3 Recuperación (`reteco.tgrag`)

Algoritmo 1 con estos parámetros:

| Parámetro | Valores | Por defecto |
|---|---|---|
| `K` (aristas top por coseno) | 20, 50, 200 | 20 (paper) |
| `edge_scope` | `query` (aristas de los candidatos de la consulta) / `pool` (todas las del dominio) | `query` |
| `graph_scope` | `subgraph` (PPR sobre `G_q`, paper) / `pool` (PPR sobre todo el grafo y se puntúan todas las aristas de los candidatos; estilo HippoRAG) | `subgraph` |
| `match` | `contained` / `overlap` / `soft` | `contained` |
| `chunk_weight` | `prod` (Π(1+γ), paper) / `none` | `prod` |
| `doc_agg` | `max` / `sum` sobre chunks | `max` |
| `fallback` | sin alcance temporal, o sin semillas en él → no se filtra | sí |

Las ablaciones de la tabla VII son *presets*:
- `full`;
- `no_ppr`: `s = 1[τ]·γ`;
- `no_temporal`: PageRank sin personalización, `s = s(v₁) + s(v₂)`;
- `no_temporal_ppr`: `s = γ`;
- `static`: τ eliminado, aristas paralelas fusionadas y `s = γ`.

**Del score TG al ranking:** el score TG solo existe para los documentos del pool (y, con `subgraph`, solo para los que contienen alguna arista de `E_q`). Se guardan dos runs:
1. el **run TG puro** (`tgrag-<ext>-<enc>/<config>`), para diagnóstico;
2. el **run fusionado** (`…+rrf<w>`): RRF del run base y el TG con peso `w`, de modo que los documentos sin score TG conservan el orden de la primera etapa.

`w` se elige en train con CV de 5 folds (como O4) y se reporta el valor de CV, no el mejor en muestra.

## 4. Experimentos

Referencia: el mejor run de primera etapa disponible en cada momento. Hoy son `bm25/rrf(limpia,titulo)` (1a, 0.153) y `bm25/titulo+paso_limpio` (1b, 0.166). Cuando esté el denso en los 13 dominios, `fusion/rrf(bm25 + bge-base)`. Criterio de aceptación: el habitual (Δ ≥ 0.01, p < 0.05, ≥ 8/13 dominios).

| ID | Experimento | Pregunta | Coste |
|---|---|---|---|
| **TG0** | Diagnóstico de cobertura (sin ranking) | % de consultas con `T_q` no vacío (reglas / oráculo); % de documentos del pool con ≥ 1 hecho fechado; % de gold con algún hecho en `T_q` frente a no-gold. Si los gold no se concentran en `T_q`, el filtro no puede ayudar → parar aquí | minutos |
| **TG1** | `full` · reglas · `T_q` reglas · `contained` · K = 20 · fusión RRF | ¿aporta el método base? | ~1 h (extracción + embeddings del pool en los 13 dominios) |
| **TG2** | Ablaciones `no_ppr`, `no_temporal`, `no_temporal_ppr`, `static` | ¿se reproduce la tabla VII? ¿es el tiempo lo que aporta (H2)? | minutos (reutiliza el grafo) |
| **TG3** | `T_q` oráculo y `T_q` LLM, frente a reglas | ¿cuánto se pierde por identificar mal el tiempo de la consulta? | oráculo: 0 · LLM: ~1 h para 1,211 consultas |
| **TG4** | `match` = `overlap` / `soft` | ¿el filtro duro descarta gold con fechas "vecinas"? (EDA 6.3) | minutos |
| **TG5** | K ∈ {20, 50, 200}; `graph_scope` = `pool`; profundidad del pool N ∈ {50, 100, 200} | ¿hace falta más cobertura que en QA? | minutos–1 h |
| **TG6** | Peso de fusión `w` ∈ {0.25, 0.5, 1, 2} con CV | robustez de la fusión | 0 |
| **TG7** | Extracción con LLM frente a reglas en `train-mini`, top-20 del pool, empezando por los dominios pequeños (`iota`, `quant`, `law`, `workplace`: 80 consultas, ~1.6 k docs) | H3: ¿compensa el coste del LLM? | ~2–4 h en local por cada 1.6 k docs |
| **TG8** | 1b: consulta `titulo+paso_limpio`; `T_q` del paso, o de la consulta madre si el paso no tiene | ¿se transfiere a 1b? | minutos |

Desgloses obligatorios en cada experimento:
- por clase e intención (H1: EAL/TPC frente a TCP/OEC);
- con `T_q` vacío frente a no vacío, porque las consultas sin alcance solo cambian por la propagación por entidades.

**Orden:** TG0 → TG1 → TG2 → TG3 (oráculo) → TG4 → TG5/TG6 → TG8 → TG7. Si TG0 o TG3-oráculo no muestran recorrido, se cierra la rama sin gastar LLM.

## 5. Código

| Archivo | Contenido | Dependencias |
|---|---|---|
| [`src/reteco/temporal.py`](src/reteco/temporal.py) | `TimeSpan`, `Scope`, extracción con regex, jerarquía, `parse_normalized`, `query_scopes`, `temporal_match` | stdlib |
| [`src/reteco/llm.py`](src/reteco/llm.py) | cliente `LLM` con caché (`outputs/llm_cache/<prompt_id>.jsonl`); backends `mlx` y `openai` (vale para LM Studio, Ollama o el endpoint compatible de Gemini) | `mlx-lm` (no instalado) u `openai` (instalado) |
| [`src/reteco/tgraph.py`](src/reteco/tgraph.py) | `Fact`, extractor de reglas, prompt y parser del extractor LLM, E/S de hechos, `TemporalGraph` (bi-nivel + índice de tiempo) | numpy, scipy |
| [`src/reteco/tgrag.py`](src/reteco/tgrag.py) | `TGConfig`, presets, PPR disperso, Algoritmo 1 adaptado, embeddings de aristas con caché incremental | numpy, scipy, `reteco.dense` (torch) |
| [`scripts/tg_pool.py`](scripts/tg_pool.py) | pools de candidatos desde runs existentes | — |
| [`scripts/tg_extract.py`](scripts/tg_extract.py) | extracción incremental de hechos (reglas o LLM) | LLM en proceso aparte |
| [`scripts/tg_query_time.py`](scripts/tg_query_time.py) | `T_q` por consulta o paso (reglas / LLM / oráculo) | — |
| [`scripts/tg_retrieve.py`](scripts/tg_retrieve.py) | grafo + embeddings + scoring + fusión + evaluación y registro en `resultados.csv` | torch (no importa `bm25`) |
| [`scripts/tg_diagnose.py`](scripts/tg_diagnose.py) | TG0: cobertura temporal de pool, consultas y gold | — |

Cambios en módulos existentes (compatibles hacia atrás):
- `data.train_mini_ids()`: el subconjunto `train-mini` de PLAN_EXPERIMENTOS §1 (40 consultas por dominio, estratificadas por clase).
- `evaluate.evaluate_variant(..., qids=None)`: permite evaluar en un subconjunto de consultas, para que los runs de `train-mini` no cuenten con 0 las consultas que faltan.

Salidas en `outputs/exp/tgrag/`:
- `pools/<pool>/<dom>__<split>.json`
- `facts/<ext>/<dom>.jsonl`
- `qtime/<método>/<subtrack>/<dom>__<split>.json`
- `edge_emb/<ext>/<enc>/<dom>.npz`

Los runs van al árbol de siempre, `outputs/exp/runs/`.

**Separación de procesos:** el LLM (mlx) va en `tg_extract.py` / `tg_query_time.py`; torch, en `tg_retrieve.py`; la JVM (BM25) no se carga en ninguno. Los pools se leen de runs ya guardados.

## 6. Cómo se ejecutará (cuando toque probar)

```bash
# Pool: top-100 del mejor run de 1a (y del denso cuando exista)
.venv/bin/python scripts/tg_pool.py --subtrack 1a --runs "bm25/rrf(limpia,titulo)" --depth 100 --name p100
# Hechos con reglas para el pool
.venv/bin/python scripts/tg_extract.py --extractor rules --pools 1a/p100
# Alcance temporal de las consultas
.venv/bin/python scripts/tg_query_time.py --subtrack 1a --method rules
.venv/bin/python scripts/tg_query_time.py --subtrack 1a --method oracle
# TG0
.venv/bin/python scripts/tg_diagnose.py --subtrack 1a --pool p100 --extractor rules --qtime rules
# TG1 + TG2
.venv/bin/python scripts/tg_retrieve.py --subtrack 1a --pool p100 --extractor rules --qtime rules \
    --base "bm25/rrf(limpia,titulo)" --presets full no_ppr no_temporal no_temporal_ppr static --rrf-weights 0.5 1
# TG7 (LLM, subconjunto)
.venv/bin/python scripts/tg_extract.py --extractor llm --pools 1a/p100 --subset mini --top 20 --domains iota quant law workplace
```

**Pendiente antes de la primera ejecución** (verificaciones, no experimentos):
1. Revisar a mano `temporal.extract_spans` con ~30 oraciones reales por dominio, para ver falsos positivos ("1500 USD", números de bloque en `bitcoin`).
2. Instalar `mlx-lm` (`uv add mlx-lm`) y confirmar el id del modelo (por defecto `mlx-community/Qwen3-4B-Instruct-2507-4bit`), o apuntar el backend `openai` a un servidor local.
3. Comprobar con `tg_diagnose.py` el tamaño del grafo de `history` (memoria: ~0.5 GB de embeddings fp16 con bge-small y pool de 100).

## 7. Riesgos y extensiones

- **El filtro temporal puede no tener recorrido en TEMPO.** El 55 % de los gold no comparte año con la consulta y muchas consultas solo tienen marcas relativas. TG0 y TG3-oráculo lo detectan pronto.
- **Las entidades de las reglas son pobres** en dominios técnicos, donde los términos clave van en minúscula ("segwit", "census"). El grafo queda poco conectado y el PPR aporta poco. TG7 mide cuánto se gana con el LLM.
- **Tope de recall:** como reranker, TG-RAG no recupera nada fuera del pool. Con `edge_scope = pool` puede traer documentos del pool de *otras* consultas del dominio; se puede explorar como extensión.
- **Extensión opcional (G1):** usar informes por nodo de tiempo (generados con LLM a partir de los hechos del pool) como contexto para HyDE o condensación de la consulta (Q1/Q3 de PLAN_EXPERIMENTOS). Solo si Q3 muestra que HyDE ayuda al denso.
- **Declaración:** si se usa el LLM en un run oficial, declarar el modelo, la versión y los prompts (están en `tgraph.py` y `tg_query_time.py`).
