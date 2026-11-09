# Plan de experimentos — Track 1 (TEMPO): tipo de razonamiento, keywords y reescritura de consultas

Objetivo: pasar del baseline BM25 (macro nDCG@10 train = 0.1075 en 1a, 0.1024 en 1b) a un pipeline que use el **tipo de razonamiento temporal** de la consulta y técnicas de **reescritura de consultas** (query rewriting, HyDE) para mejorar la primera etapa.

Este plan parte de lo encontrado en [`notebooks/EDA_EXPLICADO.md`](notebooks/EDA_EXPLICADO.md) y [`BASELINE.md`](BASELINE.md). Las reglas de trabajo no cambian: **todo se decide en train**; dev solo se usa para confirmar a los ganadores al final; nunca se usan el formato de los IDs, `gold_answers` ni `guidance` en inferencia.

---

## 0. Punto de partida: qué sabemos y qué implica para la idea propuesta

La idea inicial es: (1) clasificar la consulta según el tipo de razonamiento, (2) armar un diccionario de keywords por tipo que indique cómo se vería un documento relevante y (3) explorar HyDE y query rewriting. Antes de diseñar los experimentos, contrasté cada parte con los datos.

### 0.1 Clasificar el tipo de razonamiento: viable, pero las etiquetas son ruidosas

- Las etiquetas (`temporal_reasoning_class_primary`, `temporal_intent`) vienen de `guidance`, que **casi seguro no estará en test**. Por eso hace falta un clasificador propio, y se puede entrenar con las etiquetas de train.
- Distribución en train (clase primaria, 1,211 consultas):

  | Clase | Abrev. paper | n | BM25 nDCG@10 (EDA 6.4) |
  |---|---|---:|---:|
  | event_analysis_and_localization | EAL | 435 | 0.137 |
  | time_period_contextualization | TPC | 256 | 0.067 |
  | origins_evolution_comparative_analysis | OEC | 171 | — |
  | trends_changes_and_cross_period | TCP | 117 | — |
  | event_verification_and_authenticity | EVA | 80 | 0.129 |
  | materials_artifacts_and_provenance | MAP | 68 | 0.063 |
  | resto (SMD, HAC, CAU, vacías…) | — | 84 | — |

- **Prueba rápida (hecha para este plan):** TF-IDF (1-2 gramas) + regresión logística, CV de 5 folds en train, sobre el texto limpio de la consulta.

  | Etiqueta | Clases | Accuracy | Macro-F1 | Clase mayoritaria |
  |---|---:|---:|---:|---:|
  | Clase primaria | 7 | 0.47 | 0.33 | 0.36 |
  | Intención (normalizada) | 7 | 0.43 | 0.38 | 0.23 |

  `duration`, `when`, `period_definition` y `timeline` se reconocen razonablemente (F1 0.45–0.55). En cambio, EVA, MAP, `order` y `ongoing_status` casi no se detectan.
- Además, las etiquetas las generó un LLM y tienen ruido (intenciones que son nombres de clase, 93 granularidades distintas…). Un clasificador **no puede superar la consistencia de las propias etiquetas**.

> **Implicación:** clasificar solo vale la pena si **el tipo cambia la estrategia de búsqueda** y eso mejora el ranking. Antes de construir el clasificador hay que medir el techo con las etiquetas reales (**experimento O4, enrutamiento oráculo**). Si con etiquetas perfectas no se gana, con etiquetas predichas tampoco.

### 0.2 Diccionario de keywords: reorientarlo

- EDA 2.5–2.6: la densidad temporal de un documento **no distingue gold de no-gold** (AUC 0.57), y en algunos dominios los gold tienen *menos* años que el resto.
- EDA 6.3: solo el **45 %** de los pares (consulta, gold) comparte algún año.
- EDA 4.5 y 7.5: el problema principal es de **vocabulario del tema**, no del tiempo. El gold dice lo mismo con otras palabras.

> **Implicación:** un diccionario de expresiones temporales genéricas ("before", "century", "in 18xx") usado para expandir la consulta en BM25 probablemente **mete ruido**. Donde sí puede servir:
> 1. como **rasgos para el clasificador** (señales de la *consulta*: "how long" → `duration`, "still / currently" → `ongoing_status`);
> 2. como **pistas en el prompt** de la reescritura condicionada por clase (qué *tipo de documento* buscar: registro parroquial, catálogo de museo, notas de versión…);
> 3. como **rasgos suaves** en un reranker posterior (coincidencia de periodo), nunca como filtro duro.
>
> Y conviene **aprenderlo de los datos** (experimento O5) antes de escribirlo a mano.

### 0.3 HyDE y query rewriting: la apuesta principal

- Las consultas son largas (mediana 127 palabras, p99 673), y la revisión manual (EDA 7.5) muestra que los términos clave se diluyen. Condensar la consulta debería ayudar incluso a BM25.
- Paper TEMPO: añadir el razonamiento de un LLM **sube** a un retriever que sigue instrucciones (ReasonIR 27.3 → 41.0) pero **hunde** a BM25 (10.8 → 4–6), porque el texto extra diluye los términos. Lo mismo pasa al añadir etiquetas temporales estructuradas (*Normalized*: ReasonIR +8, el resto ≈ 0). **El efecto depende del retriever**, así que cada variante de consulta debe probarse con más de uno.
- IIMAS-RAG (SemEval-2026, [`papers/2026.semeval-1.345.pdf`](papers/2026.semeval-1.345.pdf)): la reescritura con LLM fue lo que más aportó (+16 % relativo en nDCG@10 sobre un híbrido SPLADE + denso con RRF).
- EDA 4.5: `gold_answers` se parece a los gold solo un poco más que la consulta. HyDE puede ayudar, pero hay que medir su techo (experimento O1).

### 0.4 Prerrequisito: la primera etapa

El EDA dejó claro que el cuello de botella es la primera etapa (BM25 R@1000 = 0.64; en `history`, 0.33) y que un denso pequeño sin ajustar ya duplica o triplica a BM25 (probado solo en `iota`, `law` y `quant`). Todas las técnicas de consulta se aplican **sobre** un retriever, así que primero hay que fijar retrievers de referencia para los 13 dominios (Fase 1).

---

## 1. Protocolo común a todos los experimentos

- **Métrica principal:** macro nDCG@10 en train (13 dominios). **Secundarias:** macro R@100 y R@1000 (calidad de la primera etapa) y nDCG@10 desglosado **por clase de razonamiento e intención** (usando las etiquetas de `guidance` solo para analizar).
- **Criterio para aceptar una mejora** frente a la referencia:
  - Δ macro nDCG@10 ≥ +0.01;
  - prueba pareada por consulta (bootstrap o aleatorización) con p < 0.05 sobre todas las consultas de train;
  - mejora en al menos 8 de los 13 dominios.

  No se decide nada mirando `iota`, `quant` o `law` aislados.
- **Subconjunto `train-mini` para experimentos con LLM:** hasta 40 consultas por dominio, estratificadas por clase primaria (~380 consultas, con todos los dominios representados). Se usa para filtrar ideas caras; las que pasan se corren en train completo.
- **Caché:** cada run se guarda en `outputs/exp/<exp_id>/<dominio>__<split>.pkl`, y cada generación de LLM en `outputs/llm_cache/<prompt_id>.jsonl`, con clave `(qid, hash del prompt, modelo, semilla)`. Así nada se genera dos veces.
- **Registro:** una fila por experimento en `outputs/exp/resultados.csv` (id, descripción, retriever, variante de consulta, macro nDCG@10, R@100, R@1000, Δ, p-valor, dominios que mejoran).
- **Fusión:** cuando se combinen consultas o retrievers, usar RRF (k = 60) como opción por defecto y una suma ponderada de scores normalizados como alternativa.

---

## 2. Fase 0 — Infraestructura (pasar del notebook a módulos)

Hoy el código vive en los notebooks. Para iterar rápido conviene extraerlo:

```
src/reteco/
  data.py         # cargadores: documentos, consultas, pasos, qrels, guidance; limpieza HTML
  bm25.py         # OfficialStyleBM25 (copiado del notebook) + caché de matrices
  dense.py        # encoder genérico (sentence-transformers), chunking, índice numpy/FAISS
  fusion.py       # RRF y fusión ponderada
  evaluate.py     # pytrec_eval, macro por dominio, bootstrap, test pareado, desglose por clase
  queries.py      # registro de variantes de consulta: nombre -> función(registro) -> texto(s)
  llm.py          # cliente LLM con caché en disco
scripts/
  run_experiment.py   # --exp <id> --retriever <x> --query-variant <y> --split train
```

- [ ] Extraer el código y comprobar que reproduce exactamente 0.1075 / 0.1024 (prueba de regresión).
- [ ] Ejecutar lo denso en **scripts aparte, sin Pyserini**, para evitar el conflicto JVM + torch/FAISS visto en el EDA.

---

## 3. Fase 1 — Retrievers de referencia en los 13 dominios

Sin esto no se puede interpretar ningún experimento de consulta.

| ID | Experimento | Detalle | Coste aprox. (M2 Pro, MPS) |
|---|---|---|---|
| R1 | BM25 con consulta limpia y con título | ya hecho en el EDA → solo registrar | 0 |
| R2 | `bge-base-en-v1.5` + chunking | trozos de 256–384 tokens, solape de 64, score del documento = max de sus trozos | ~1.5–3 h para codificar el corpus completo |
| R3 | `e5-base-v2` (o `gte-base`) + chunking | mismo protocolo, para no depender de un único modelo | ~1.5–3 h |
| R4 | Encoder que acepta instrucciones (p. ej. `Qwen3-Embedding-0.6B`) | permite probar el efecto *Normalized* del paper (etiquetas de clase en la instrucción); empezar por los dominios pequeños | ~8 h para todo el corpus; decidir tras probar en dominios pequeños |
| R5 | Híbrido RRF(BM25, mejor denso) | el EDA vio que solo ayudaba en `iota`; confirmarlo en los 13 | 0 |

**Salida:** 1 o 2 retrievers de referencia (probablemente un denso + BM25 como brazo léxico) sobre los que se ejecuta todo lo demás.

---

## 4. Fase 2 — Oráculos: medir el techo antes de construir

Usan información de train que no estará en test (`gold_answers`, `guidance`). **No son sistemas**: sirven para decidir qué ramas merecen trabajo. No necesitan LLM, así que son baratos.

| ID | Pregunta | Variante de consulta | Si sale bien… | Si sale mal… |
|---|---|---|---|---|
| O1 | ¿Cuánto podría dar HyDE con una respuesta "perfecta"? | `gold_answers` como consulta; consulta + `gold_answers` (RRF y concatenación) | HyDE tiene recorrido: priorizar la Fase 5 | Un pseudo-documento no se parece a los gold: bajar la prioridad de HyDE y subir la de condensación |
| O2 | ¿Ayuda dar el tiempo de forma estructurada? (*Normalized* del paper) | consulta + `INTENT=…; CLASS=…; ANCHORS=[…]; EVENTS=[…]` tomado de guidance | Vale la pena extraer anclas y eventos con LLM o reglas | Las etiquetas no aportan al retriever: el clasificador solo serviría para enrutar |
| O3 | ¿Cuánto da una reescritura tipo LLM? | `query_summary`; `query_summary` + `key_time_anchors` + `query_temporal_events`; acciones del `retrieval_plan` (RRF entre acciones) | Techo de la Fase 5 con un LLM similar | — |
| O4 | **¿Merece la pena clasificar?** | para cada clase, elegir la mejor variante (de R*, O1–O3 y las variantes del EDA); comparar "enrutamiento con etiqueta real" frente a "mejor variante única para todos" | Se construye el clasificador (Fase 3) | **Descartar el clasificador**; usar la estrategia única ganadora |
| O5 | ¿Hay keywords de documento específicas de cada clase? | por clase, log-odds (prior de Dirichlet) de términos en gold frente a hard negatives (top-50 de BM25 no gold de la misma consulta) | Semilla del diccionario (Fase 4) | El diccionario solo sirve como pistas de prompt, no como expansión |

**Cuidados:**
- `guidance` lo generó un LLM, posiblemente viendo los gold (el `retrieval_plan` trae `gold_ids` adjuntos). O2 y O3 pueden sobreestimar lo alcanzable: tratarlos como **cotas superiores**, no como estimaciones.
- O4 se calcula con validación cruzada (elegir la mejor variante por clase en 4 folds y evaluar en el 5.º) para no sobreajustar la elección.

---

## 5. Fase 3 — Clasificador del tipo de razonamiento (solo si O4 lo justifica)

- [ ] **Normalizar etiquetas.** Clases: las 6 frecuentes + `otra`. Intención: `when`, `period_definition`, `timeline`, `ongoing_status`, `duration`, `order` (`before_after` / `after` / `since`) + `otra`. Mover a `otra` los valores de intención que en realidad son nombres de clase.
- [ ] Modelos, de más barato a más caro:

  | ID | Modelo | Referencia |
  |---|---|---|
  | C1 | TF-IDF + regresión logística | **ya medido:** acc 0.47 / macro-F1 0.33 (clase), 0.43 / 0.38 (intención) |
  | C2 | Reglas sobre la consulta (diccionario de señales, §6.1) + C1 | se espera mejorar `duration`, `ongoing_status` y `order` |
  | C3 | Embedding de la consulta (el mismo encoder de R2/R4) + regresión logística | |
  | C4 | LLM zero-shot / few-shot con las definiciones de la taxonomía | útil también para predecir **varias** clases a la vez |

- [ ] **Evaluación:** F1 por CV, pero sobre todo **nDCG@10 aguas abajo**: repetir el enrutamiento de O4 con etiquetas predichas. Si la ganancia del enrutamiento oráculo desaparece con etiquetas predichas, se usa la estrategia única.
- [ ] Probar **enrutamiento suave**: en lugar de elegir una estrategia, fusionar todas pesándolas por la probabilidad de cada clase. Es más robusto con un clasificador de ~50 % de acierto.

---

## 6. Fase 4 — Diccionario de keywords

### 6.1 Dos diccionarios distintos

Conviene separarlos porque sirven para cosas diferentes:

| Clase / intención | Señales en la **consulta** (para clasificar) | Pistas de **documento** (para prompts y rasgos) |
|---|---|---|
| EAL / `when` | "when did", "what year", "date of", "at what point" | fecha concreta + nombre del evento; "in 18xx", "on <fecha>" |
| TPC / `period_definition` | "during the", "era", "period", "at the time", "back then" | "century", "era", "contemporary", descripciones de la época |
| OEC | "origin", "first", "earliest", "how did X become", "evolve" | "originated", "first used", "derived from", "introduced in" |
| TCP | "changed", "over time", "compared to", "nowadays", "used to" | "increased", "declined", "by 19xx", "since then", series por periodos |
| EVA | "is it true", "really", "myth", "authentic", "did X actually" | "according to", "evidence", "records show", "hoax", "disputed" |
| MAP | "how old", "made of", "this artifact / photo / coin", "dating" | "manufactured", "hallmark", "circa", "material", "maker" |
| SMD | "where can I find", "records", "source", "archive" | "register", "census", "archive", "catalogue", "transcription" |
| CAU | "why", "what led to", "caused", "reason" | "because", "as a result", "due to", "led to" |
| `ongoing_status` | "still", "currently", "as of", "nowadays", "anymore" | "as of 20xx", "currently", "latest version", "deprecated" |
| `duration` | "how long", "for how many", "lasted" | "for N years", "until", "lasted" |

Además, cada dominio tiene sus **marcadores temporales propios**, que conviene añadir: versiones y BIPs en `bitcoin`/`monero`/`cardano`/`iota`; fechas de sentencias y enmiendas en `law`; años de censo y registros parroquiales en `genealogy`; mandatos y legislaturas en `politics`.

La tabla de la derecha es una **hipótesis**: hay que contrastarla con O5 y quedarse solo con lo que aparece en los datos.

### 6.2 Experimentos

| ID | Experimento | Hipótesis / expectativa |
|---|---|---|
| K1 | Expansión genérica: añadir a la consulta BM25 los términos temporales de su clase | Control negativo: se espera ≈ 0 o negativo (EDA 2.5). Confirmarlo y cerrar la rama |
| K2 | Expansión con términos de O5 (aprendidos de train, con CV), con peso bajo | Si O5 encontró términos discriminativos, pequeña mejora en BM25 |
| K3 | Pistas de documento por clase en la **instrucción** del encoder R4 ("Retrieve a museum catalog entry or provenance record that dates…") | Equivalente realista de O2 |
| K4 | Normalización de anclas: extraer años de la consulta y añadir su década y siglo ("1834" → "1830s", "19th century"); boost suave a documentos con años en ±N | Ayuda en EAL, TPC y `genealogy`; usar solo como señal suave (EDA 6.3: el 55 % de los gold no comparte año) |

---

## 7. Fase 5 — Reescritura de consultas con LLM

Cada variante se prueba con **BM25 y con el mejor denso**, sola y fusionada (RRF) con la consulta original. Primero en `train-mini`; lo que pase el criterio, en train completo.

| ID | Técnica | Qué genera el LLM | Cómo se usa |
|---|---|---|---|
| Q1 | **Condensación** | 1 consulta corta (≤ 20 palabras) con entidades, evento y periodo | sustituye a la consulta o se fusiona con ella; ataca la dilución en consultas largas |
| Q2 | **Multi-query** | 3–5 reformulaciones diversas (sinónimos, vocabulario del dominio) | RRF entre todas; ataca el desajuste de vocabulario |
| Q3 | **HyDE** | 1–3 pasajes hipotéticos estilo Wikipedia o foro del dominio | denso: media de embeddings (consulta + pasajes); BM25: estilo query2doc (consulta ×3 + pasaje) |
| Q4 | **Extracción estructurada** | JSON con `{evento, entidades, anclas temporales, periodo, intención}` | la versión realista de O2: plantilla *Normalized* + insumo para K4 |
| Q5 | **Reescritura condicionada por clase** | prompt específico por clase (predicha en la Fase 3), con las pistas de §6.1 | combina la idea original: clasificación → keywords → rewriting |
| Q6 | **Descomposición por periodo** (para TCP / OEC) | una subconsulta por periodo que haya que comparar | recuperar por separado y fusionar intercalando, para cubrir todos los periodos (TCP es la clase más difícil del paper) |
| Q7 | **Razonamiento + consulta** (estilo paper, fig. 8) | razonamiento paso a paso sobre qué evidencia hace falta | solo con encoders que sigan instrucciones (R4); con BM25 se espera que empeore |

**Para 1b** (los pasos ya son una reescritura hecha por los organizadores):
- **S1:** recuperar con el paso y con la consulta limpia por separado y fusionar (decisión del EDA 5.3).
- **S2:** convertir el paso imperativo en pregunta o en una consulta sin verbos de instrucción ("identify", "consult"…) con un LLM o con reglas.
- **S3:** HyDE por paso.
- **S4:** usar el ranking de 1a de la misma consulta como prior: los gold de 1b son siempre un subconjunto de los de 1a.

**Orden sugerido:** Q1 → Q3 → Q2 (las tres no dependen del clasificador) → Q4 → Q5 / Q6 (solo si O4 y la Fase 3 lo justifican) → Q7 (solo si existe R4).

**Elección del LLM.** Las reglas permiten modelos abiertos y APIs comerciales, siempre que se declaren en el *system description* (modelo, versión y prompts).

Volumen estimado (consulta limpia ≈ 250 tokens + prompt ≈ 300; salida de 100–300 tokens):

| Unidad | Llamadas | Tokens aprox. |
|---|---:|---:|
| 1 variante, 1a, train + dev | 1,730 | ~1.3 M |
| 1 variante, 1b (por paso), train + dev | 3,976 | ~2.5 M |
| Fase 5 completa (Q1–Q7, S2–S3, con iteraciones de prompt) | ~20–30 k | ~15–25 M |

El límite que importa es el de **peticiones por día**, no el de tokens.

- **Decisión: LLM local como motor principal.** Una familia instruct sin modo de razonamiento (Qwen 4–8B, o Gemma como alternativa), cuantizada a 4 bits, con `mlx-lm`. Calcular ~5–8 s por llamada: ~1 h para `train-mini` y ~3–4 h por variante en train + dev. No usar los modelos destilados de DeepSeek-R1: generan cadenas de razonamiento largas que multiplican el tiempo.
- **API gratuita (Gemini) solo para comparar calidad:** misma variante sobre `train-mini` con el modelo local y con Gemini. Si la diferencia en nDCG@10 es grande, valorar una API de pago para las variantes ganadoras.
- Fijar el modelo y su versión, y guardar todas las generaciones en `outputs/llm_cache/`, para que los resultados sean reproducibles y se puedan declarar.

---

## 8. Cronograma tentativo

| Semana | Trabajo | Entregable |
|---|---|---|
| 1 | Fase 0 + R1–R3, R5 | módulos en `src/reteco/`, runs densos de los 13 dominios, `resultados.csv` |
| 2 | Oráculos O1–O5 | decisión documentada: ¿clasificador sí o no?, ¿HyDE o condensación?, ¿diccionario? |
| 3 | Q1–Q3 en `train-mini` → train; S1–S2 para 1b; K1/K4 | mejores variantes de consulta por retriever |
| 4 | Fase 3 (si procede) + Q4–Q6; R4 si se eligió | pipeline 1a/1b completo; **primera y única validación en dev** de los ganadores |

Después vendrían el reranker cross-encoder (hard negatives de train) y la diversificación de casi-duplicados, que ya estaban en el pipeline propuesto por el EDA y quedan fuera de este plan.

---

## 9. Preguntas abiertas

1. ~~¿Permiten las reglas usar LLMs externos?~~ **Resuelta:** sí, tanto modelos abiertos como APIs comerciales, declarándolos ([participate.html](https://datascienceuibk.github.io/RETECO/participate.html)). Lo que no se permite es usar colecciones de recuperación externas ni la web en un run oficial.
2. ¿El test traerá algún campo de `guidance`? Si trajera la clase o la intención, la Fase 3 sobra.
3. Presupuesto de cómputo y tiempo: ¿se puede usar una GPU externa (Colab, cluster) para R4 y para codificar `history` y `hsm`?
