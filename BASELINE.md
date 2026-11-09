# Baseline de RETECO Track 1 (TEMPO) y enfoques del paper

Este documento reúne dos cosas:

1. Cómo funciona el baseline oficial que se reproduce en [`notebooks/baseline_track1_tempo.ipynb`](notebooks/baseline_track1_tempo.ipynb) (secciones 1 y 2).
2. Qué enfoques de razonamiento temporal evalúa el paper de TEMPO ([`papers/2601.09523v1.pdf`](papers/2601.09523v1.pdf)) y qué implican para nuestro pipeline.

---

## Parte 1 — Cómo funciona el baseline, paso a paso

### Idea general

El baseline oficial (`starter_kit/official_baseline.py`) es un **buscador léxico BM25**. Puntúa cada documento según las palabras que comparte con la consulta. Las palabras raras pesan más, y un documento no gana solo por ser largo. No entiende el significado: si la consulta dice *car* y el documento *automobile*, no hay coincidencia.

**No hace ningún razonamiento temporal.** Un año como *2020* es un término más. No lo relaciona con *2019* ni con *last year*, y no entiende expresiones como "antes de" o "después de".

En el notebook, el baseline se reproduce con la clase `OfficialStyleBM25`. Esta clase copia la fórmula del oficial, pero usa matrices dispersas de scipy, que gastan menos memoria que gensim con los 200k documentos de `history`. Sus detalles técnicos son:

- el **analizador de Lucene** de Pyserini: minúsculas, sin *stopwords*, *stemming* de Porter;
- la fórmula de `gensim.LuceneBM25Model` con `k1 = 0.9` y `b = 0.4`, que **pondera también la consulta** con BM25;
- la consulta **cruda, con su HTML**, en 1a, y `f"{query}\n\nStep: {step_instruction}"` en 1b;
- top-1000 documentos por consulta.

Fórmulas:

$$w(t,d) = \text{idf}(t)\,\frac{tf_{t,d}}{tf_{t,d} + k_1\,(1 - b + b\,|d|/\overline{|d|})},\qquad \text{idf}(t)=\ln\!\Big(1+\frac{N-df_t+0.5}{df_t+0.5}\Big)$$

$$\text{score}(q,d) = \sum_{t \in q \cap d} w(t,q)\cdot w(t,d)$$

Los números `PASO n` de abajo coinciden con los comentarios `# PASO n` del código del notebook.

### A. Indexar el corpus (una vez por dominio; caché en `outputs/eda/bm25/`)

1. **Analizar y contar.** Cada documento pasa por el analizador de Lucene: minúsculas → quitar *stopwords* (*the*, *of*…) → *stemming* de Porter (*halving*, *halved* → *halv*). Se cuenta cuántas veces sale cada término (`tf`). Con eso se construye a mano una matriz dispersa CSR de documentos × términos.
   - Ejemplo: `"The Bitcoin <b>halving</b> in 2020"` → `["bitcoin", "b", "halv", "b", "2020"]`.
   - Ojo: el analizador **no limpia el HTML**. Restos de etiquetas como `b` o `href` se cuelan como términos.
2. **Estadísticas del dominio.**
   - `N`: número de documentos.
   - `|d|`: longitud de cada documento, en términos analizados.
   - `avgdl`: longitud media.
   - `df_t`: en cuántos documentos aparece cada término. De ahí sale el `idf`.

   Un término que sale en casi todos los documentos, como *bitcoin* en el dominio bitcoin, tiene un idf cercano a 0 y casi no cuenta.
3. **Peso BM25 de cada par documento-término.** Aquí actúan dos efectos:
   - **Saturación** (`k1 = 0.9`): pasar de 1 a 2 apariciones suma mucho, pero pasar de 20 a 21 casi nada.
   - **Penalización por longitud** (`b = 0.4`): los documentos largos pierden un poco de peso.
4. **Guardar la matriz transpuesta** (términos × documentos) en formato disperso, porque casi todos sus valores son cero. Al buscar solo hacen falta las filas de los pocos términos de la consulta, y en CSR eso es muy rápido.

### B. Buscar (para cada consulta)

5. **Convertir la consulta en vector.** Se analiza igual que un documento y se cuentan sus términos. Los términos que no existen en el corpus del dominio se descartan.
   - Peculiaridad del oficial (gensim): **la consulta también recibe peso BM25**, con idf incluido. La longitud de la consulta hace el papel de `|d|`. En consultas largas, como los posts con HTML, las palabras repetidas saturan y las raras pesan más.
6. **Puntuar todos los documentos** con `score(q,d) = Σ peso_consulta(t) · peso_doc(t)`, sumando sobre los términos que comparten consulta y documento.
   - En el código es un único producto matriz-vector: `WT[idx].T @ w`.
   - Un documento que no comparte ningún término con la consulta obtiene score 0.
7. **Top-1000.**
   - `argpartition` encuentra los k mejores sin ordenar todo el corpus.
   - Después se ordenan por score descendente. Los empates se resuelven por el orden en el corpus, igual que el `sorted` estable del oficial.
   - El resultado es un *run* `{consulta: {doc: score}}`, que se guarda en `outputs/eda/runs/`.

### C. Evaluar

8. **Por consulta**, con `pytrec_eval`, la misma librería que el scorer oficial:
   - **nDCG@10** (métrica oficial): mide si los documentos gold están en las primeras posiciones. Penaliza de forma logarítmica: un gold en la posición 1 vale más que en la 5.
   - **Recall@10/100/1000**: mide qué parte de los gold aparece entre los k primeros resultados, sin importar el orden. Sirve para saber si el problema está en la primera etapa (el gold ni siquiera aparece) o en el orden (aparece, pero abajo, y un reranker podría subirlo).
9. **Macro-media**: primero la media por dominio y luego la media de los 13 dominios, de modo que todos pesan igual. En la sección 2.1 del notebook se comprueba que estas cifras coinciden con las de `starter_kit/BASELINE_RESULTS.md`.

### Dónde está cada paso en el notebook

- **Pasos 1–7:** celda de `lucene_analyze` y `OfficialStyleBM25` (sección 1).
- **Pasos 8–9:** celda de `evaluate_run` y `macro_by_domain`.
- **Ejecución:** celda de `TOPIC_VARIANTS` / `run_bm25_variants`. Solo `1a_oficial` y `1b_oficial` son el baseline; las demás variantes (`1a_limpia`, `1a_titulo`, `1b_paso`, `1b_consulta`, `1b_limpia+paso`) son experimentos para comparar. Los resultados por consulta quedan en `EVAL["1a_oficial"]` y `EVAL["1b_oficial"]`.

### Mini-ejemplo

Consulta: *"bitcoin halving 2020 price"* → tras analizar: `[bitcoin, halv, 2020, price]`.

- *bitcoin* está en casi todos los documentos del dominio: idf cercano a 0, no ayuda.
- *halv* y *2020* son raros: idf alto, son los que deciden.
- Un documento corto que dice "halving" dos veces y "2020" una puntuará alto.
- Uno muy largo que dice "price" cien veces apenas sube, por la saturación.

### Por qué es un baseline flojo para Track 1 (tempo)

- No limpia el HTML, que mete ruido en la consulta.
- Trata los años como palabras sueltas: *2020* no coincide con *2019* ni con *last year*.
- No capta sinónimos ni paráfrasis.

---

## Parte 2 — Enfoques del paper de TEMPO para el razonamiento temporal

El paper de TEMPO **no propone un método temporal propio**. Es un **benchmark**: construye el dataset y las métricas, y mide 12 recuperadores que ya existen. Lo que hace es sobre todo diagnóstico.

### 1. Los modelos que comparan (tabla 3, nDCG@10 medio)

| Familia | Modelos | nDCG@10 |
|---|---|---|
| Léxico (*sparse*) | BM25 | **10.8** |
| Densos pequeños (<1B) | BGE, Contriever, Inst-L, SBERT | 21–25 |
| Densos grandes (>1B) | E5-Mistral, GritLM, Qwen, SFR | 23–30 |
| *Reasoning-enhanced* | **DiVeR**, ReasonIR, Rader | 24–**32.0** |

- **Densos:** codifican la consulta y cada documento en un vector y buscan por similitud. Captan sinónimos y paráfrasis, pero ninguno se diseñó pensando en el tiempo.
- **Reasoning-enhanced:** recuperadores entrenados para consultas que exigen razonar, no solo encontrar las mismas palabras. El paper casi no explica cómo funcionan por dentro.

**DiVeR gana, pero por poco:** 32.0 frente a 30.4 de E5 y 30.0 de SFR. No es un modelo temporal. Viene de otro paper (Long et al., 2025, *"Diver: A multi-stage approach for reasoning-intensive information retrieval"*, arXiv 2508.07995), pensado para el benchmark BRIGHT.

Lo siguiente no está en el paper de TEMPO; viene del paper original de DiVeR. DiVeR es un pipeline de varias etapas:

1. Limpia los documentos.
2. Expande la consulta de forma iterativa con un LLM.
3. Recupera con un embedding entrenado con ejemplos sintéticos y *hard negatives*.
4. Reordena los resultados con un LLM.

TEMPO no aclara si ejecutaron el pipeline completo o solo el recuperador. Por la sección 6, donde dicen que DiVeR tiene el razonamiento "internalizado", parece que usaron solo el recuperador, pero no lo afirman.

Diferencias por dominio:
- History: DiVeR 34.5.
- Politics: 47.9 (E5).
- Law: 40.8 (SFR).
- Iota: 41.7 (ReasonIR).
- Monero, el más difícil: 23.7 (SFR).

### 2. Las métricas temporales (sección 4, tabla 4)

nDCG no dice si los documentos encontrados cubren los periodos de tiempo que pide la consulta. Por eso añaden cuatro métricas, y en todas **un LLM hace de juez**: decide si cada documento es temporalmente relevante y a qué periodo pertenece.

- **TP@k (Temporal Precision):** precisión temporal ponderada por posición. Premia que los documentos temporalmente relevantes salgan arriba.
- **TR@k (Temporal Relevance):** fracción del top-k que es temporalmente relevante.
- **TC@k (Temporal Coverage):** fracción de los periodos requeridos que cubre al menos un documento. Usan 2 periodos: el de referencia y el de comparación.
- **nDCG|FC@k:** nDCG calculado solo en las consultas donde se cubren todos los periodos.

| Modelo | TP@10 | TR@10 | TC@10 | nDCG\|FC@10 |
|---|---|---|---|---|
| BM25 | 24.0 | 11.1 | 32.8 | 22.5 |
| DiVeR | **62.0** | **41.3** | 71.4 | 32.5 |
| ReasonIR | 57.4 | 38.2 | **72.4** | 28.8 |
| E5 | 53.5 | 31.1 | 63.2 | **33.4** |

Estas métricas no se pueden calcular con `pytrec_eval`, porque necesitan anotaciones temporales que nosotros no tenemos. **La métrica oficial de la tarea sigue siendo nDCG@10.**

### 3. Los enfoques que prueban para el aspecto temporal

#### a) Variantes de la consulta (tabla 5)

Es lo más cercano a "usar el tiempo" en el paper.

| Variante | Qué hace | Resultado |
|---|---|---|
| *Strip* | Quita fechas y expresiones temporales | Cae solo unos 2 puntos de media; BM25 casi no cambia |
| *Temporal-Only* | Deja solo la información temporal | Se hunde (DiVeR: 32.0 → 17.7). El tiempo sin el tema no sirve |
| *Normalized* | Añade etiquetas, p. ej. `TEMPORAL_INTENT=when; ANCHORS=[2015, 2023]` | ReasonIR sube a **35.3 (+8)** y supera a DiVeR; al resto casi no le afecta |

Conclusión: los modelos actuales apenas aprovechan las señales temporales tal como vienen en el texto. En cambio, si se les da el tiempo **estructurado y explícito**, un modelo que sigue instrucciones como ReasonIR sí lo usa.

Nos interesa porque los archivos `guidance_*.jsonl` traen justo ese tipo de metadatos (intención y anclas temporales). Es lo que se analiza en la sección 5 de `notebooks/eda_track1_tempo.ipynb`.

#### b) Reformulación con LLM (figura 8)

Un LLM (GPT-4o, Llama-70B, Qwen-72B, Qwen-32B, DeepSeek-32B) escribe un razonamiento paso a paso, que se añade a la consulta.

- ReasonIR pasa de 27.3 a **41.0** con GPT-4o. Es el mejor número de todo el paper.
- BM25 empeora (10.8 → 4–6), porque el texto extra diluye las palabras clave.
- DiVeR se queda igual (30.7–32.2), como si ya razonara por dentro.

#### c) Descomposición en pasos, la tarea 1b (figura 6)

| Estrategia | Qué se usa como consulta | nDCG@10 medio |
|---|---|---|
| *Step-Only* | Solo el paso | 14.6 |
| *Query+Step* | Consulta + paso (**lo que hace el baseline oficial de 1b**) | 26.4 |
| *Query+All* | Consulta + todos los pasos | 25.9 |

- Un paso aislado no tiene contexto suficiente.
- ReasonIR es el que más gana (17.2 → 35.0–35.3).
- DiVeR pasa de 20.1 a 33.3 / 32.0.
- BM25 no cambia (10.3–11.6).
- Recomiendan **mantener siempre la consulta original junto al paso**.

#### d) Análisis por clase de razonamiento temporal (figura 7)

Usan 10 clases: EAL, TPC, OEC, TCP, EVA, MAP, OTH, SMD, CAU, HAC.

- La más difícil es **TCP** (tendencias y comparación entre periodos): DiVeR solo llega a 23.9.
- La más fácil es **HAC** (atribución histórica y contexto): SFR llega a 51.8.
- DiVeR es el más equilibrado: gana en 6 de las 10 clases.
- ReasonIR destaca en **CAU** (causalidad): 28.1.

#### e) RAG (tabla 6)

Generador: Llama-3-70B; evaluador: GPT-4o (puntuación 0–100).

| Recuperador | Puntuación |
|---|---|
| Sin recuperación | 77.3 |
| Oracle (docs gold) | **80.5** |
| DiVeR | 75.6 |
| BGE | 74.8 |
| BM25 | 73.8 |

**No recuperar nada funciona mejor que recuperar con cualquier modelo.** Su hipótesis es que los documentos que no cubren todos los periodos confunden al generador.

---

## Qué significa para nosotros

1. **Hay mucho margen sobre BM25.** Un denso decente ya duplica su nDCG@10.
2. Lo que más rinde en el paper es **un recuperador que sigue instrucciones y recibe la información temporal de forma explícita**: etiquetas de intención y anclas (hasta 35.3) o un razonamiento generado por un LLM (hasta 41.0).
3. **En 1b hay que mantener siempre la consulta junto al paso.**
4. **La clase TCP (comparar periodos) es el punto débil de todos los modelos**, así que es buen candidato a recibir un tratamiento específico. Por ejemplo, recuperar por separado para cada periodo y fusionar los resultados.

> **Comparación con el paper:** las cifras del paper (BM25 = 10.8) se evaluaron sobre las 1,730 consultas (train + dev) y sobre el corpus original sin deduplicar; las de `starter_kit/BASELINE_RESULTS.md` son solo train y corpus v1.1. La sección 3 de `notebooks/baseline_track1_tempo.ipynb` las compara: las consultas, los pasos y el tamaño del corpus (v1.1 + eliminados) coinciden exactamente con el paper, y el macro de nuestro baseline en train + dev (11.0 en 1a) está a 0.2 puntos del paper. Por dominio hay diferencias de ~2.5 puntos de media (hasta 11 en `iota`, que tiene 10 consultas), así que úsalas como tendencias y compara solo en macro.
