# Plan de experimentos — SPLADE (recuperación *sparse* aprendida) para el Track 1 (TEMPO)

Objetivo: comprobar si un retriever *sparse* aprendido (SPLADE) mejora la primera etapa de 1a y 1b, **solo** o, sobre todo, **como brazo léxico del híbrido**, en lugar de BM25 o junto a él.

El plan parte de lo medido en [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) y de la revisión del repositorio de IIMAS-RAG para SemEval-2026 Task 8 ([github.com/Vania-Janet/mtrag_semeval2026](https://github.com/Vania-Janet/mtrag_semeval2026)), cuyo paper ya tenemos en [`papers/2026.semeval-1.345.pdf`](papers/2026.semeval-1.345.pdf). Las reglas no cambian: **todo se ajusta en train**, dev solo se usa al final para confirmar a los ganadores, y en inferencia nunca se usan el formato de los IDs, `gold_answers` ni `guidance`.

**Estado (2026-11-18):** el código de §3 está implementado y probado en el Mac (pruebas y humo en `iota`, §9). En el servidor no se ha ejecutado nada y el código aún no está subido.

---

## 0. Punto de partida

Macro nDCG@10 en train (13 dominios):

| Referencia | 1a | 1b |
|---|---:|---:|
| Baseline oficial (BM25) | 0.1075 | 0.1024 |
| Mejor BM25 | 0.1528 · `rrf(limpia,titulo)` | 0.1660 · `titulo+paso_limpio` |
| Mejor denso `bge-base` (mejor trozo) | 0.2277 · `rrf(limpia,titulo)` | 0.2449 · `limpia+paso` |
| **Híbrido actual** (RRF, BM25 con w = 0.5) | **0.2367** | **0.2610** |

Lo que ya sabemos y condiciona este plan:

- **BM25 y el denso piden consultas distintas.** BM25 mejora al acortar la consulta (el título solo gana a la consulta completa). El denso, en cambio, quiere la consulta completa y limpia.
- **El brazo BM25 solo suma si pesa la mitad que el denso** (w = 0.3–0.6). Con el mismo peso mete ruido arriba del ranking, porque es mucho peor (0.15 frente a 0.23).
- **El problema principal es de vocabulario, no de tiempo** (EDA 4.5 y 7.5): el gold dice lo mismo con otras palabras. Además, solo el 45 % de los pares consulta–gold comparte algún año (EDA 6.3).

---

## 1. Qué es SPLADE y por qué probarlo en TEMPO

### 1.1 Cómo funciona

SPLADE pasa el texto por un BERT con su cabeza de *masked language modeling* (MLM). Esa cabeza da, para cada token de entrada, un logit por cada término del vocabulario (30,522 en BERT *uncased*). El vector del texto es:

$$w_j = \max_{i \in \text{tokens}} \log\big(1 + \text{ReLU}(\text{logit}_{i,j})\big)$$

- Sale un vector de 30,522 dimensiones con unos cientos de valores distintos de cero. Es *sparse*, como el de BM25, y se busca con un índice invertido o con un producto de matrices dispersas.
- Los pesos de los términos son **aprendidos**, no salen de `tf` e `idf`.
- Hay **expansión**: se activan términos que no están en el texto (sinónimos, palabras relacionadas). Esto ataca justo el desajuste de vocabulario.
- El score es el producto escalar entre el vector de la consulta y el del documento.
- Los términos activos son legibles, así que se puede **ver qué expansiones hace** el modelo, algo que no se puede hacer con un denso.

### 1.2 Por qué puede ayudar aquí

1. **Desajuste de vocabulario.** Es el problema principal del EDA, y la expansión de SPLADE lo ataca directamente. A la vez conserva el emparejamiento exacto de nombres propios, entidades y años, que es donde los densos fallan más.
2. **Un brazo léxico mejor para el híbrido.** Hoy el brazo léxico (BM25) es tan flojo que hay que darle la mitad de peso. Si SPLADE se acerca al denso, la fusión puede ganar más, porque los dos brazos serían buenos y distintos.
3. **Evidencia externa.** En MT-RAG, IIMAS-RAG midió SPLADE como **el mejor modelo individual** (0.480, frente a 0.440 de `bge-base` y 0.250 de BM25). Su sistema final fue el híbrido SPLADE + denso con RRF (§2.3).
4. **Hueco en TEMPO.** El paper de TEMPO evalúa BM25, densos y *reasoning-enhanced*, pero **ningún retriever sparse aprendido**. Sería una aportación propia para el *system description*.

### 1.3 Riesgos

- **Cambio de dominio.** Los modelos SPLADE públicos se entrenaron en MS MARCO, con consultas de ~6 palabras y pasajes cortos. Nuestras consultas tienen una mediana de 127 palabras (p99: 673), y los documentos son páginas web y posts.
- **Consultas largas.** Una consulta larga activa cientos de términos. La búsqueda es más lenta y entra ruido, igual que con BM25. Por eso hay un experimento de poda de la consulta (SP3).
- **Truncado.** El modelo lee como mucho 512 tokens. Los documentos se trocean igual que en el denso (§3.2).
- **Años.** Hay que comprobar cómo tokeniza BERT los años (un token o varios *wordpieces*), porque eso determina si SPLADE empareja fechas exactas (SP5).
- **Licencia.** Los modelos de NAVER (`splade-cocondenser-ensembledistil`, `splade-v3`…) son CC BY-NC-SA 4.0. El uso no comercial en una tarea compartida es compatible, pero hay que declararlo. Hay alternativas Apache 2.0 si hiciera falta (§3.1).

---

## 2. Lo que aprendimos del repo de IIMAS-RAG

Se revisó el código de `TASK A/` (`src/pipeline/indexing/indexer.py`, `src/pipeline/retrieval/sparse.py`, `fusion.py`, `run.py`, `configs/`), sus resúmenes estadísticos y el paper.

### 2.1 Cómo corren SPLADE: lo que sí nos sirve

| Aspecto | Qué hacen | Qué hacemos nosotros |
|---|---|---|
| Modelo | `naver/splade-cocondenser-ensembledistil` (SPLADE++ ED). Lo confirman el código y la tabla 5 del paper | Lo usamos como **réplica** (`splade-ed`) y lo comparamos con `splade-v3` (SP2) |
| Codificación | `AutoModelForMaskedLM`; `log(1 + relu(logits)) * attention_mask`; *max-pooling* sobre los tokens | La misma fórmula, vía `SparseEncoder` de sentence-transformers (§3.2). Una prueba comprueba que coinciden |
| Simetría | La misma función codifica consultas y documentos, sin prefijos | Igual (`encode_query` / `encode_document` no añaden prefijos en estos modelos) |
| Longitud | `max_length = 512` con truncado, para consultas y documentos | 512 para consultas; documentos en trozos de 254 tokens (§3.2) |
| Índice | Matriz dispersa scipy CSR (documentos × vocabulario) en `index.npz`, más `doc_ids.json` | Lo mismo por dominio, en `outputs/exp/splade/<modelo>/<dominio>.npz` |
| Búsqueda | Producto disperso matriz × vector de consulta, búsqueda exacta, top-300 por consulta | Producto disperso exacto por lotes de consultas, top-1000 (lo necesitan R@1000 y la fusión) |
| Poda | Ninguna ("full sparse activation") | Ninguna en documentos; la poda de la consulta se prueba aparte (SP3) |
| Fusión | RRF con k = 60 y pesos uniformes. RRF supera a la fusión lineal (+0.043, p = 0.015) | RRF k = 60, pero **con pesos** (en nuestro híbrido los pesos importaron) |

### 2.2 Lo que no hay que copiar

- **La memoria del indexador.** `SpladeIndexer` acumula **vectores densos** de 30,522 floats por documento y solo convierte a CSR al final. Son ~122 KB por trozo: `history` (330 k trozos) necesitaría ~40 GB, y el corpus completo, ~230 GB. Ellos tenían 128 GB de RAM y corpus más pequeños. **Nosotros convertimos a disperso en cada batch.**
- **`DataParallel` con dos GPUs.** Reúne en `cuda:0` los logits de tamaño batch × tokens × 30,522, que es justo el tensor más pesado. Con dos 3070 de 8 GB conviene **un proceso independiente por GPU** (`CUDA_VISIBLE_DEVICES`) con los dominios repartidos.
- **fp32 en la codificación.** El indexador SPLADE no usa fp16. En una 3070, fp16 hace falta para que quepa el batch y para la velocidad.
- **El README no coincide con el código.** El README dice "SPLADE-v3" y da cifras más altas (0.577 de media) que las del paper (0.528). Nos fiamos del código y del paper.
- **No trocean.** Los corpus de MT-RAG ya vienen en pasajes (`passage_level_processed`). Los nuestros son documentos largos, así que necesitamos el troceo del denso.
- Su BM25 es `rank_bm25` (Okapi, k1 = 1.2, b = 0.75), no el de Lucene. Sus comparaciones con BM25 no se trasladan a las nuestras.

### 2.3 Resultados que orientan el plan

Tabla 1 del paper: dev de MT-RAG, nDCG@10 medio de 4 dominios. "Std" = el último turno como consulta.

| Sistema | nDCG@10 |
|---|---:|
| BM25 (Std) | 0.250 |
| `bge-base-en-v1.5` (Std) | 0.440 |
| Voyage-3-large (Std) | 0.440 |
| BGE-M3 (Std) | 0.460 |
| **SPLADE-ED (Std)** | **0.480** |
| SPLADE + `bge-base` RRF (Std) | 0.435 |
| SPLADE + Voyage RRF (Std) | 0.454 |
| SPLADE + Voyage RRF + HyDE / + multi-query | 0.383 / 0.401 |
| SPLADE + `bge-base` RRF + reescritura | 0.481 |
| SPLADE + Voyage RRF + reescritura (final) | 0.528 |
| ELSER + reescritura (baseline de los organizadores; otro sparse aprendido) | 0.540 |

Lecturas para nosotros:

- **SPLADE fue el mejor modelo individual**, por delante de densos del tamaño de nuestro `bge-base`. El mejor número de la tabla también es de un sparse aprendido (ELSER).
- **Fusionar con pesos iguales puede empeorar**: SPLADE + `bge-base` sin reescritura (0.435) queda por debajo de SPLADE solo (0.480). Es lo mismo que nos pasó con BM25 a w = 1. Hay que barrer pesos.
- **HyDE y multi-query empeoraron**, y la reescritura de la consulta fue lo que más aportó (+0.189 frente al último turno). Encaja con nuestro O1 (HyDE con la respuesta gold apenas aporta) y con la Fase 5 del plan general.
- **El reranker** `bge-reranker-v2-m3` sobre el top-100 empeoró (−0.029 nDCG@5) y no lo usaron.

---

## 3. Implementación

Se sigue el diseño de `dense.py`, para que `run_experiment.py`, `fuse_runs.py` y `evaluate.py` sirvan sin cambios. Todo el código lleva docstrings y comentarios en español, como el resto del proyecto.

### 3.1 Archivos

| Archivo | Contenido |
|---|---|
| `src/reteco/splade.py` (nuevo) | registro `MODELS`, carga del modelo, `encode_corpus`, `SpladeIndex` |
| `scripts/encode_splade.py` (nuevo) | `--model splade-ed --domains …`: codifica el corpus; se puede interrumpir y relanzar |
| `scripts/run_experiment.py` | `open_index` reconoce los retrievers `splade-*` (con `@first` / `@maxpool` y poda `-qN`, ver §3.3) |
| `scripts/compare_runs.py` (nuevo) | complementariedad de dos runs (SP1): gold que solo encuentra uno, recall de la unión, Jaccard del top-10 |
| `tests/test_splade.py` (nuevo) | pruebas de §3.4 |

Modelos del registro:

| Clave | Modelo HF | Notas |
|---|---|---|
| `splade-ed` | `naver/splade-cocondenser-ensembledistil` | réplica de IIMAS; libre de descarga |
| `splade-v3` | `naver/splade-v3` | más reciente; **gated**: hay que aceptar los términos en HF y tener un token en el servidor |
| `splade-v3-db` | `naver/splade-v3-distilbert` | más rápido; solo si el tiempo importara |
| `os-v2` | `opensearch-project/opensearch-neural-sparse-encoding-v2-distill` | Apache 2.0; solo si la licencia NC fuera un problema |

En el servidor ya está `SparseEncoder` (sentence-transformers 6.2), con `encode_query`, `encode_document` y `max_active_dims`.

### 3.2 Codificación del corpus

- **Troceo idéntico al denso:** trozos de 254 tokens, solape de 30, como mucho 4 por documento (los ~1000 primeros tokens), con `reteco.dense.chunk_texts`. Si el tokenizador de SPLADE tiene el mismo vocabulario WordPiece que `bge-base` (se comprueba en SP0), los trozos son **los mismos** que los del denso. Así, `max` y `first` se comparan uno a uno.
- **fp16 en CUDA**, con los trozos ordenados por longitud de mayor a menor. Si el batch no cabe, falla en los primeros minutos y no a mitad. Empezar con batch 32.
- **Disperso en cada bloque.** Se codifica en bloques de ~20 k trozos con `convert_to_sparse_tensor=True` y cada bloque pasa a CSR de scipy. **Nunca** se materializa la matriz densa trozos × 30,522.
- **Salida:** `outputs/exp/splade/<modelo>/<dominio>.npz` con la CSR (`data` en fp16, `indices`, `indptr`), `chunk_doc` y `doc_ids`. El script imprime los términos activos por trozo (media, p50, p99) y el tiempo.
- **Dos GPUs:** dos procesos en `tmux`, uno por GPU, con los dominios repartidos a ~50 % de los trozos cada uno (§6).

### 3.3 Búsqueda (`SpladeIndex`)

- Carga la CSR del dominio y guarda su **transpuesta** (vocabulario × trozos). Es un índice invertido: cada fila es la lista de trozos de un término.
- Codifica las consultas por lotes, con un máximo de 512 tokens y con poda opcional a los `N` términos de más peso (`max_active_dims`).
- Por lotes de 64 consultas: `S = Q @ Dᵀ` (disperso × disperso, en CPU) y después `S` pasa a denso (64 × trozos).
- Agrega por documento:
  - `max`: el mejor trozo (`np.maximum.reduceat`, como en el denso);
  - `first`: el primer trozo;
  - `maxpool`: el máximo de los vectores de los trozos de cada documento, término a término. Equivale a un único vector por documento que cubre sus ~1000 primeros tokens. Es específico de SPLADE y sale de la misma codificación.
- Top-k con `fusion.top_k`, con el mismo desempate que el oficial.
- **Nombres de retriever:** `splade-ed`, `splade-ed@first`, `splade-ed@maxpool`, `splade-ed-q64` (consulta podada a 64 términos), etc. Los runs van, como siempre, a `outputs/exp/runs/<retriever>/<variante>/`.
- Usa torch, así que no se importa en el mismo proceso que `reteco.bm25` (JVM), igual que el denso.

### 3.4 Pruebas (`tests/test_splade.py`)

1. La conversión a CSR por bloques da la misma matriz que la fórmula densa, sobre ~50 textos.
2. `SparseEncoder` da los mismos vectores que la fórmula manual del repo (`log1p(relu(logits)) · mask`, máximo sobre los tokens) para `splade-ed`, con tolerancia de fp16.
3. Las agregaciones `max`, `first` y `maxpool` coinciden con un cálculo por fuerza bruta en un índice de juguete.
4. El desempate de top-k es el del oficial.

### 3.5 Recursos

| Concepto | Valor | Origen |
|---|---|---|
| Documentos / trozos a codificar | 1.167 M / **1.874 M** | contado en `outputs/exp/dense/bge-base/*.npz`* |
| Tiempo de codificación por modelo | ~1–3 h con las dos 3070 | estimación (`bge-base` tardó ~4.6 h en el MPS del Mac); SP0 lo mide |
| Tamaño del índice por modelo | **~2–3 GB** | `splade-ed` en `iota`: 196 términos activos por trozo de media (p99: 315) → 1.874 M × ~200 × 6 bytes |
| RAM al buscar (dominio mayor, `history`) | ~1–2 GB | matriz y transpuesta en float32; caben de sobra en 60 GB |
| Memoria de GPU | logits de 32 × 256 × 30,522 en fp16 ≈ 0.5 GB, más intermedios | dentro de 8 GB |
| Disco libre en el servidor | 809 GB | medido |

\* [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) §2.7 habla de 3.47 M trozos, pero los `.npz` suman 1.874 M.

---

## 4. Experimentos

**Protocolo:** el de [`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md) §1.

- Métrica principal: macro nDCG@10 en train. Secundarias: R@100, R@1000 y el desglose por clase e intención.
- Una mejora se acepta con Δ ≥ +0.01, p < 0.05 (test pareado de aleatorización) y mejora en ≥ 8 de 13 dominios.
- Cada run va a `resultados.csv`. Dev no se toca.

**Referencias:**

- (a) el mejor BM25, para saber si SPLADE es mejor brazo léxico;
- (b) el mejor denso;
- (c) **el híbrido actual (w = 0.5)**, que es la base que hay que batir.

| ID | Experimento | Coste | Depende de |
|---|---|---|---|
| SP0 | Montaje y prueba de humo en `iota`, `quant` y `law` | minutos de GPU | — |
| SP1 | `splade-ed` solo en 13 dominios: variantes de consulta × agregación | 1 codificación + CPU | SP0 |
| SP2 | Elección del modelo: `splade-v3` frente a `splade-ed` | 1 codificación + CPU | SP1 |
| SP3 | Poda de la consulta (top-N términos) | CPU, sin recodificar | SP2 |
| SP4 | **Fusión:** SPLADE + denso, y BM25 + SPLADE + denso | CPU | SP3 |
| SP5 | Diagnóstico: expansiones, años, complementariedad, desglose por clase | CPU | SP1–SP4 |
| SP6 | Confirmación en dev | — | al final, junto con las demás ramas |

### SP0 · Montaje y prueba de humo (no se decide nada)

- Codificar `iota`, `quant` y `law` (~109 k trozos) con `splade-ed` en una GPU.
- **Medir:**
  - trozos por segundo, para calcular el tiempo del corpus completo;
  - términos activos por trozo y por consulta;
  - tamaño del índice y RAM;
  - segundos por consulta en la búsqueda.
- **Comprobar:**
  - las pruebas de §3.4;
  - que los trozos coinciden con los del denso;
  - que SPLADE supera a BM25 en esos tres dominios.

  Lo último es solo una **comprobación de cordura**. Si SPLADE quedara por debajo de BM25, lo primero sería buscar un error (falta el ReLU o el log, el *pooling* está mal, el truncado o los IDs están mal) antes de seguir.
- Por la regla del protocolo, **no se decide nada** con estos tres dominios.

### SP1 · SPLADE solo en los 13 dominios

- Codificar los 13 dominios con `splade-ed`, dos procesos en `tmux` (§6).
- **Variantes:**
  - 1a: `cruda`, `limpia`, `titulo`, `titulo+inicio`, `rrf(limpia,titulo)`;
  - 1b: `oficial`, `paso`, `limpia+paso`, `titulo+paso`, `titulo+paso_limpio`, `rrf(paso,limpia)`, `rrf(titulo+paso,limpia)`.
- **Agregación:** `max`, `first` y `maxpool`. Las tres salen de la misma codificación.
- **Hipótesis:**
  - **H1.** SPLADE supera al mejor BM25 en los dos subtracks según el criterio de aceptación. Si no lo hace (ni con `splade-v3` en SP2), **se cierra la rama**.
  - **H2.** Su consulta preferida queda entre la de BM25 (corta) y la del denso (completa). Que gane `limpia` o `titulo` dice si su ponderación aprendida resiste el ruido de las consultas largas.
- **Complementariedad con el denso**, la medida que predice si la fusión puede ganar:
  - fracción de gold en el top-100 de SPLADE que no están en el top-100 del denso, y al revés;
  - solapamiento (Jaccard) de los top-10.

  Si SPLADE apenas encuentra gold propios, SP4 tendrá poco margen.

### SP2 · Elección del modelo

- Codificar con `splade-v3` (antes hay que dar acceso, §8) y comparar con `splade-ed` en las **2 mejores variantes** de SP1 por subtrack, con la mejor agregación.
- **Regla:** `splade-v3` solo sustituye a `splade-ed` si lo supera según el criterio de aceptación. En caso de duda se queda `splade-ed`: replica a IIMAS y no está restringido.
- `splade-v3-db` y `os-v2` no se prueban, salvo que el tiempo o la licencia lo exijan.

### SP3 · Poda de la consulta

- Con BM25, acortar la consulta fue lo que más rindió. SPLADE permite acortarla **por importancia** (los N términos de más peso) en lugar de con una heurística (el título), y sin recodificar el corpus.
- N ∈ {32, 64, 128, 256, sin poda}, sobre la mejor variante de cada subtrack. Primero se mira la distribución de términos activos por consulta, para que la rejilla tenga sentido.
- Se adopta N solo si mejora a la versión sin poda según el criterio. Es además más rápido.

### SP4 · Fusión (el experimento principal)

Con `fuse_runs.py` (RRF, k = 60) y la mejor configuración de SP1–SP3. La referencia es **el híbrido actual** (1a 0.2367, 1b 0.2610).

| ID | Fusión | Rejilla (peso del denso = 1) |
|---|---|---|
| F1 | SPLADE + denso (sustituye a BM25) | w_SPLADE ∈ {0.3, 0.5, 0.7, 1, 1.5} |
| F2 | BM25 + SPLADE + denso | w_BM25 ∈ {0.25, 0.5} × w_SPLADE ∈ {0.5, 1} |
| F3 | Solo 1b: varias variantes SPLADE + denso | p. ej. SPLADE `titulo+paso_limpio` + SPLADE `limpia+paso` + denso `limpia+paso` |

- **Cómo se fija el peso:** igual que en R5, el valor redondo dentro de la meseta, no el máximo exacto. Es la protección contra el sobreajuste en train.
- **Se informa siempre de R@1000.** Es el techo del futuro reranker; aunque el nDCG@10 empate, un híbrido con más recall puede valer como generador de candidatos.
- **Ablación de k de RRF** (20, 60, 100): solo si F1 o F2 ganan, para comprobar que la elección es estable.

### SP5 · Diagnóstico (sin decisiones; para el informe y para fases posteriores)

1. **Expansiones.** Para 30 consultas estratificadas por clase:
   - los 20 términos de más peso de la consulta, marcando cuáles son expansión y cuáles están en el texto;
   - si aparecen términos temporales (años, décadas, `century`, `era`) que no estaban en la consulta.

   Esto conecta con K4 (normalización de anclas): quizá SPLADE ya la hace de forma implícita.
2. **Años en el vocabulario.** Cuántos años de 1000 a 2030 son un único token en el vocabulario de BERT, y qué peso reciben en la consulta y en los gold que los contienen.
3. **Consulta sin expansión.** Se ponen a cero los términos que no están en la consulta. Separa cuánto aporta la ponderación aprendida y cuánto la expansión. Es barato: no hay que recodificar.
4. **Desglose por clase e intención** (EAL, TPC, `when`, `period_definition`…) frente al denso y al híbrido, como en TG-RAG. Responde si el emparejamiento exacto de SPLADE ayuda en las clases con fechas concretas.
5. **Ejemplos:** 5 gold que encuentra SPLADE y no el denso, y 5 al revés.

### SP6 · Dev

Solo con la configuración ganadora y **una sola vez**, junto con las demás ramas, al final (regla del proyecto).

---

## 5. Árbol de decisión

```
SP0 ── ¿SPLADE ≥ BM25 en iota/quant/law? ── no ──► buscar el error antes de seguir
  │ sí
SP1/SP2 ── ¿mejor SPLADE > mejor BM25 (criterio) en 1a y 1b?
  │                        └── no ──► cerrar la rama; documentar (SP5 resumido)
  │ sí
SP4 ── ¿F1 o F2 > híbrido actual (criterio)?
  │ sí ──► nueva base de la primera etapa; actualizar RESULTADOS y PLAN_EXPERIMENTOS
  │ no ──► ¿sube R@1000 sin bajar nDCG@10? ── sí ──► guardar como generador de candidatos
  │                                         └── no ──► cerrar la rama (SPLADE solo como resultado del informe)
```

---

## 6. Orden de ejecución y comandos

Antes de ejecutar nada en el servidor hay que subir el código nuevo y los cambios pendientes del Mac (commit + push, **cuando lo indiques**). Hoy el servidor está en `54c0e5b`.

```bash
# Servidor: actualizar el código
ssh myserver 'cd ~/RETECO && git pull --ff-only'

# SP0: prueba de humo, una GPU (minutos)
ssh myserver 'cd ~/RETECO && CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/encode_splade.py --model splade-ed --domains iota quant law'
ssh myserver 'cd ~/RETECO && .venv/bin/python scripts/run_experiment.py --subtrack 1a --retriever splade-ed --variants limpia "rrf(limpia,titulo)" --domains iota quant law'

# SP1: resto del corpus, una sesión tmux por GPU (~50 % de los trozos en cada una)
#   GPU 0: history, travel, bitcoin, monero, cardano        (~865 k trozos)
#   GPU 1: hsm, genealogy, politics, economics, workplace   (~900 k trozos)
ssh myserver 'cd ~/RETECO && tmux new -d -s splade_g0 "CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/encode_splade.py --model splade-ed --domains history travel bitcoin monero cardano > outputs/exp/logs/splade_ed_g0.log 2>&1"'
ssh myserver 'cd ~/RETECO && tmux new -d -s splade_g1 "CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/encode_splade.py --model splade-ed --domains hsm genealogy politics economics workplace > outputs/exp/logs/splade_ed_g1.log 2>&1"'
ssh myserver 'tail -3 ~/RETECO/outputs/exp/logs/splade_ed_g*.log'

# SP1: variantes (CPU; reutiliza los runs ya calculados)
ssh myserver 'cd ~/RETECO && .venv/bin/python scripts/run_experiment.py --subtrack 1a --retriever splade-ed --variants cruda limpia titulo titulo+inicio "rrf(limpia,titulo)" --ref "bm25/rrf(limpia,titulo)"'
ssh myserver 'cd ~/RETECO && .venv/bin/python scripts/run_experiment.py --subtrack 1b --retriever splade-ed --variants oficial paso limpia+paso titulo+paso titulo+paso_limpio "rrf(paso,limpia)" "rrf(titulo+paso,limpia)" --ref bm25/titulo+paso_limpio'
#   …lo mismo con --retriever splade-ed@first y splade-ed@maxpool

# SP1: complementariedad con el denso (solo describe; no escribe en resultados.csv)
ssh myserver 'cd ~/RETECO && .venv/bin/python scripts/compare_runs.py --subtrack 1a --runs "splade-ed/rrf(limpia,titulo)" "bge-base/rrf(limpia,titulo)"'

# SP4: fusión (ejemplo para 1a, con la mejor variante SPLADE de SP1–SP3)
ssh myserver 'cd ~/RETECO && .venv/bin/python scripts/fuse_runs.py --subtrack 1a --runs "splade-ed/rrf(limpia,titulo)" "bge-base/rrf(limpia,titulo)" --weights 0.5 1 --ref "fusion/rrf(bm25:rrf(limpia,titulo)+bge-base:rrf(limpia,titulo))w0.5,1"'
```

Calendario orientativo:

| Paso | Trabajo | Tiempo |
|---|---|---|
| 1 | Implementación y pruebas (Mac) | ~½ día |
| 2 | SP0 | ~1 h |
| 3 | SP1: codificación + variantes + complementariedad | 1–3 h de máquina + análisis |
| 4 | SP2: codificación de `splade-v3` | 1–3 h de máquina |
| 5 | SP3 + SP4 | ~1–2 h de CPU |
| 6 | SP5 y redacción de `RESULTADOS_SPLADE.md` (más el resumen en `RESULTADOS_EXPERIMENTOS.md` §2.10) | ~½ día |

---

## 7. Fuera de alcance (por ahora)

- **Ajustar (fine-tune) SPLADE con los qrels de train.** Con 1,211 consultas el riesgo de sobreajuste es alto, y antes hay que saber si el modelo sin ajustar sirve. Si SP4 gana con holgura, sería el siguiente paso natural, con validación cruzada en train.
- **Sparse de BGE-M3** (`lexical_weights`). IIMAS lo indexó pero no lo publica. Es un XLM-R large (568 M parámetros), unas 5 veces más caro de codificar.
- **Índice de impacto de Pyserini/Lucene.** El producto exacto con scipy cabe en memoria y evita mezclar la JVM con torch.
- **Reescritura con LLM.** Es la Fase 5 de [`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md). Si SPLADE entra en el híbrido, cada reescritura se probará también sobre el brazo SPLADE (en IIMAS fue lo que más aportó).
- **Repetir O4** (enrutamiento) con SPLADE como candidato cualitativamente distinto. Estaba pendiente desde la Fase 2 y se hace después de SP4.

---

## 8. Preguntas abiertas

1. **Acceso a `splade-v3`.** Está restringido (*gated*, con aprobación automática). Hay que aceptar los términos en su página de Hugging Face con tu cuenta y dejar un token en el servidor (`hf auth login`, o `HF_TOKEN` en el entorno del `tmux`). Como es una credencial, lo tienes que hacer tú. Sin token, SP2 se sustituye por `splade-v3-db` (libre) o se omite.
2. **Licencia CC BY-NC-SA 4.0** de los modelos de NAVER. Se puede usar en una tarea compartida académica, pero hay que declararla en el *system description*. Si se prefiere evitarla, existe `os-v2` (Apache 2.0).
3. **Cambios pendientes en el Mac** (los comentarios de 20+ archivos aún sin commit). ¿Se suben en un commit aparte antes del código de SPLADE?

---

## 9. Registro de implementación

### 2026-11-18 · Código de §3, probado en el Mac

**Archivos.** [`src/reteco/splade.py`](src/reteco/splade.py), [`scripts/encode_splade.py`](scripts/encode_splade.py), [`scripts/compare_runs.py`](scripts/compare_runs.py) y [`tests/test_splade.py`](tests/test_splade.py) son nuevos; en [`scripts/run_experiment.py`](scripts/run_experiment.py) solo cambian `open_index` y el docstring. El denso, BM25, la fusión y la evaluación no se tocan.

**Decisiones de implementación:**

- **Codificación con `SparseEncoder`** de sentence-transformers. Una prueba comprueba que da los mismos vectores que la fórmula manual del repo de IIMAS.
- **Conversión a CSR por bloques** de 20 k trozos, ordenados de más largo a más corto.
- **Guardado atómico** en `.npz`, con valores en float16.
- **Búsqueda:** producto disperso exacto contra la transpuesta (el índice invertido), en lotes de 64 consultas.
- **`-qN` y `-noexp` (SP3, SP5):** primero se aplica la máscara y después la poda, con código propio en lugar de `max_active_dims`, para controlar ese orden.

**Pruebas:** `.venv/bin/python tests/test_splade.py` da 9/9 (con `RETECO_SPLADE_REAL=1`, que descarga `splade-ed`); `tests/test_extraction.py` sigue en 16/16.

- Ocho pruebas usan un BERT diminuto aleatorio:
  - fórmula, bloques y orden;
  - guardado y lectura en float16;
  - agregaciones frente a fuerza bruta;
  - empates;
  - poda y `-noexp`;
  - de extremo a extremo;
  - nombres de los retrievers.
- La del modelo real confirma:
  - `SparseEncoder` = fórmula manual con `splade-ed`;
  - el tokenizador de `splade-ed` tiene **el mismo vocabulario que `bge-base`**;
  - `chunk_texts` da **los mismos trozos** con los dos.

**Humo en `iota`** (Mac, MPS, fp16; salida en el *scratchpad*, sin tocar `outputs/` ni `resultados.csv`):

- 9,680 documentos → 16,637 trozos, el mismo número por documento que `bge-base`.
- 196 términos activos por trozo de media (p50: 197; p99: 315).
- 81 trozos por segundo en el MPS (el ritmo en la 3070 se medirá en SP0), 3.4 min en total y 0.02 GB.
- Cada variante se busca en menos de 1 s.

nDCG@10 de cordura. **`iota` tiene solo 7 consultas en 1a y 16 pasos en 1b: no se decide nada con esto.**

| Variante | SPLADE `max` | SPLADE `first` | SPLADE `maxpool` | BM25 | `bge-base` |
|---|---:|---:|---:|---:|---:|
| 1a `limpia` | 0.248 | 0.197 | 0.299 | 0.098 | 0.211 |
| 1a `rrf(limpia,titulo)` | 0.157 | 0.123 | 0.115 | 0.143 | 0.210 |
| 1b `limpia+paso` | 0.157 | 0.136 | 0.167 | 0.125 | 0.250 |
| 1b `titulo+paso_limpio` | 0.164 | 0.156 | 0.158 | 0.182 | 0.168 |

SPLADE supera a BM25 con la misma consulta en 1a `limpia` y en 1b `limpia+paso`, que era la comprobación de cordura de SP0. La tubería funciona de punta a punta, incluido `open_index` con nombres como `splade-ed-q64-noexp@maxpool`.

`compare_runs.py`, aplicado a BM25 `limpia` frente a `bge-base` `limpia` (1a), reproduce los R@100 de [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) (0.359 y 0.617).

**Pendiente:**

- Commit y push (cuando se indique) y SP0 en el servidor.
- El script de diagnóstico de SP5 (expansiones, años), que se escribirá cuando haya resultados de SP1.
