# Resultados de los experimentos — baseline, BM25, denso, híbrido, clasificación del razonamiento, K4 y perfil temporal (estado al 2026-11-21)

Resumen de lo implementado de [`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md) y de los resultados obtenidos. El detalle del perfil temporal implícito con n-gramas está en [`RESULTADOS_PERFIL_TEMPORAL.md`](RESULTADOS_PERFIL_TEMPORAL.md). Todas las cifras son **macro nDCG@10 en train** (13 dominios), salvo que se indique otra cosa. Dev no se ha tocado.

**Estado:**

| Fase | Estado |
|---|---|
| 0 · Infraestructura | ✅ terminada; reproduce exactamente el baseline oficial |
| 1 · Retrievers de referencia | ✅ BM25, denso `bge-base` (13 dominios) e híbrido R5. **Mejor primera etapa:** híbrido BM25 + denso |
| 2 · Oráculos | ✅ O1–O3 con BM25 y con el denso; O4 (repetido con BM25, denso e híbrido en §2.11); O5 |
| 3 · Clasificador del tipo de razonamiento | ❌ **cerrada** (§2.4, §2.11): ni con etiquetas perfectas el enrutamiento supera al híbrido |
| 4 · Keywords | ❌ K1/K2 descartadas (§2.5); K4 cerrada (§2.9) |
| K4 · décadas y siglos de las fechas de la consulta | ✅ §2.9. ❌ **No ayuda** en ningún brazo ni en el híbrido: se cierra |
| TT0 · perfil temporal implícito con n-gramas (*focus time*) | ✅ §2.10 (detalle en [`RESULTADOS_PERFIL_TEMPORAL.md`](RESULTADOS_PERFIL_TEMPORAL.md)). Señal débil (AUC ≈ 0.6) solo en los dominios históricos; no se adopta |
| 5 · Reescritura con LLM | fuera del alcance del proyecto |

---

## 1. Qué se creó

### Módulos (`src/reteco/`)

| Archivo | Contenido |
|---|---|
| [`data.py`](src/reteco/data.py) | rutas, cargadores (consultas, pasos, qrels, guidance), limpieza HTML, normalización de etiquetas (`clase`: 6 clases + `other`; `intent`: 6 intenciones + `other`) |
| [`bm25.py`](src/reteco/bm25.py) | `OfficialStyleBM25` del notebook, reutilizando los índices de `outputs/eda/bm25/` |
| [`dense.py`](src/reteco/dense.py) | encoders (`bge-small`, `bge-base`, `e5-base`) en fp16 sobre MPS; chunking de 254 tokens, solape de 30 y máximo 4 trozos por documento; agregación `max` (mejor trozo) o `first` (primer trozo) |
| [`fusion.py`](src/reteco/fusion.py) | `top_k` con el desempate del oficial y RRF (también por consulta con varios textos) |
| [`queries.py`](src/reteco/queries.py) | registro de variantes de consulta de 1a y 1b, incluidas las de oráculo, `strip_step` (quita con regex el verbo imperativo y las palabras meta de los pasos) y las de K4 (`+k4`, `rrf(...,k4)` y sus controles `k4ctl`) |
| [`temporal.py`](src/reteco/temporal.py) | expresiones temporales con regex (años, meses, décadas, siglos, rangos, a. C.) y su papel (cita, metadato); `coarse_forms` da la década y el siglo de las fechas de un texto (K4) |
| [`evaluate.py`](src/reteco/evaluate.py) | `pytrec_eval`, macro por dominio, test pareado de aleatorización (sobre la macro), desglose por clase e intención, registro en `outputs/exp/resultados.csv` |

`bm25` (JVM) y `dense` (torch) se importan solo en procesos distintos, para evitar la caída del kernel que se vio en el EDA.

### Scripts (`scripts/`)

| Script | Uso |
|---|---|
| [`run_experiment.py`](scripts/run_experiment.py) | `--subtrack 1a --retriever bm25 --variants all`: corre, evalúa y compara con el baseline oficial; reutiliza los runs ya calculados |
| [`encode_corpus.py`](scripts/encode_corpus.py) | `--model bge-base`: codifica los corpus por trozos; se puede interrumpir y relanzar |
| [`fuse_runs.py`](scripts/fuse_runs.py) | RRF de runs ya calculados (p. ej. el híbrido BM25 + denso, R5) |
| [`oracle_routing.py`](scripts/oracle_routing.py) | O4: mejor estrategia única frente al enrutamiento por clase, intención o dominio (con CV; en 1b, folds por consulta madre) y frente al oráculo por consulta |
| [`oracle_class_terms.py`](scripts/oracle_class_terms.py) | O5: log-odds de términos de los gold frente a los hard negatives, por clase |
| [`k4_diag.py`](scripts/k4_diag.py) | K4: cobertura y comparación base → K4 en todo train, con fecha / sin fecha y por dominio |
| [`temporal_profile_diag.py`](scripts/temporal_profile_diag.py) | TT0: léxico n-grama → periodo, perfiles temporales y separación gold / no-gold en el pool del híbrido |

### Salidas (`outputs/exp/`)

- `runs/<retriever>/<variante>/<dominio>__train.pkl`: rankings top-1000.
- `resultados.csv`: una fila por experimento (nDCG@10, R@10/100/1000, Δ, p, dominios que mejoran / empeoran).
- `oracles/`: tablas de O4 y O5.
- `dense/bge-base/`: embeddings por dominio.
- `perfil_temporal/`: léxicos y tablas de TT0.
- `logs/`: registros de ejecución.

---

## 2. Resultados

### 2.1 Regresión del baseline

| | Oficial | Nuestro |
|---|---:|---:|
| 1a | 0.1075 | **0.1075** |
| 1b | 0.1024 | **0.1024** |

### 2.2 Variantes de consulta con BM25 (sin LLM, realistas)

Δ y p frente al baseline oficial; "dominios" = cuántos de los 13 mejoran / empeoran.

**1a**

| Variante | nDCG@10 | R@100 | R@1000 | Δ | p | Dominios |
|---|---:|---:|---:|---:|---:|---|
| `cruda` (oficial) | 0.1075 | 0.336 | 0.640 | — | — | — |
| `limpia` | 0.1126 | 0.359 | 0.661 | +0.005 | 0.51 | 11/1 |
| `titulo+inicio` (título + 64 palabras) | 0.1347 | 0.408 | 0.686 | +0.027 | <0.001 | 12/1 |
| `titulo` | 0.1500 | 0.423 | 0.688 | +0.042 | <0.001 | 11/2 |
| **`rrf(limpia,titulo)`** | **0.1528** | **0.453** | **0.723** | **+0.045** | <0.001 | **12/1** |

**1b**

| Variante | nDCG@10 | R@100 | R@1000 | Δ | p | Dominios |
|---|---:|---:|---:|---:|---:|---|
| `oficial` | 0.1024 | 0.392 | 0.701 | — | — | — |
| `paso` | 0.1104 | 0.407 | 0.658 | +0.008 | 0.41 | 9/4 |
| `limpia+paso` | 0.1107 | 0.416 | 0.719 | +0.008 | 0.008 | 11/2 |
| `rrf(paso,limpia)` | 0.1432 | 0.502 | 0.768 | +0.041 | <0.001 | 13/0 |
| `titulo+paso` | 0.1611 | 0.529 | 0.779 | +0.059 | <0.001 | 13/0 |
| **`titulo+paso_limpio`** | **0.1660** | **0.544** | **0.806** | **+0.064** | <0.001 | **13/0** |

**Lectura:** con BM25, **acortar la consulta es lo que más rinde**. El título solo supera a la consulta completa, y fusionar los dos es aún mejor. En 1b, sustituir la consulta larga por el título y quitar al paso el verbo imperativo y las palabras meta ("primary sources", "official documentation") mejora en los 13 dominios. Ya es una mejora de +0.045 / +0.064 sin ningún modelo.

### 2.3 Oráculos (cotas superiores; usan datos que no estarán en test)

| ID | Variante (1a, BM25) | nDCG@10 | Δ vs oficial | Lectura |
|---|---|---:|---:|---|
| O1 | `respuesta` (HyDE "perfecto") | 0.0971 | −0.010 | **Con BM25, HyDE no sirve ni con la respuesta real**: un texto largo diluye los términos, como en el paper (BM25 10.8 → 4–6) |
| O1 | `limpia+respuesta` | 0.0981 | −0.009 | ídem |
| O1 | `rrf(limpia,respuesta)` | 0.0952 | −0.012 | sube R@1000 (0.696) pero baja el top-10 |
| O2 | `normalizada` (INTENT/CLASS/ANCHORS) | 0.1128 | +0.005 (n.s.) | **las etiquetas temporales estructuradas no ayudan a BM25**, igual que en el paper |
| O2 | `limpia+anclas` | 0.1141 | +0.007 (n.s.) | ídem |
| O3 | `resumen` (query_summary de un LLM) | 0.1380 | +0.031 | una reescritura condensada ayuda… |
| O3 | `titulo+resumen` | 0.1470 | +0.039 | …pero **no supera al título solo (0.1500)** |
| O3 | `rrf(limpia,plan)` | 0.1550 | +0.048 | ≈ `rrf(limpia,titulo)` (0.1528): el plan del LLM aporta +0.002 |

**Lectura:** para BM25, el techo de "reescritura con LLM" que dan los textos de `guidance` apenas supera la heurística del título. Lo que importa es **quitar ruido** a la consulta, no añadir texto. Estas conclusiones valen para BM25; el paper indica que los encoders densos, sobre todo los que siguen instrucciones, reaccionan distinto. O1–O3 con el denso están en §2.7.

### 2.4 O4 · ¿Sirve clasificar la consulta para elegir la estrategia?

Candidatos: las variantes realistas de BM25 de §2.2. Enrutamiento con etiquetas **reales** de guidance y CV de 5 folds.

| | 1a por clase | 1a por intención | 1b por clase |
|---|---:|---:|---:|
| Mejor estrategia única | 0.1528 | 0.1528 | 0.1660 |
| Enrutamiento (CV) | 0.1436 (Δ −0.009, p 0.14) | 0.1472 (Δ −0.006, p 0.43) | 0.1667 (Δ +0.001, p 0.36) |
| Oráculo por consulta | 0.2201 | 0.2201 | 0.2413 |

- **Ni con etiquetas perfectas el enrutamiento por clase mejora.** La misma variante gana en casi todas las clases (el título en 1a, `titulo+paso_limpio` en 1b, en las 7 clases).
- El oráculo por consulta (0.22) muestra que **sí hay variación entre consultas, pero el tipo de razonamiento no la explica**.
- **Matiz:** las candidatas son todas léxicas y parecidas entre sí. Hay que repetir O4 cuando haya estrategias cualitativamente distintas (denso, HyDE, descomposición por periodo). De momento, la Fase 3 (clasificador) **no está justificada**.
- En 1b los folds se agrupan por consulta madre, para que los pasos de una misma consulta no queden repartidos entre la elección y la prueba (§2.11).
- **Repetido con BM25, denso e híbrido, y con pesos de fusión por grupo, en §2.11.** La conclusión es la misma, y la Fase 3 queda cerrada.

### 2.5 O5 · ¿Hay keywords de documento propias de cada clase?

Log-odds (prior de Dirichlet) de términos en gold frente a hard negatives (top-50 de BM25, no gold) por clase. Controles:
- primeras 150 palabras de cada documento, porque sin truncar todo salía sesgado por la longitud;
- solo términos presentes en gold de ≥ 5 consultas;
- solo negativos de ID corto, para no medir el artefacto del ID.

- **Las pistas manuales del plan (§6.1) no se sostienen.** De 37 términos, solo "century" en TPC es significativo (z = 3.5). El resto tiene |z| < 2, y "museum" en MAP sale negativo (−2.4).
- Los términos "específicos" de cada clase son **temáticos** (`troops`, `soviet`, `utxo`, `cuba`…): reflejan los dominios y los temas de las consultas, no el tipo de razonamiento.
- **La única señal fuerte es la misma en todas las clases:**
  - los hard negatives son **páginas bibliográficas o editoriales** (`copyright`, `reserved`, `press`, `university`, `library`, `www`, `book`);
  - los gold son **texto narrativo o enciclopédico**.

**Decisión:** descartar K1/K2 (expansión con keywords por clase). La señal "bibliografía / boilerplate" se apunta como **rasgo de calidad del documento**, para penalizar suavemente el score. Es un rasgo de contenido, no del formato del ID.

### 2.6 Denso (preliminar, solo `iota` y `quant`)

| Config (2 dominios, 31 consultas) | nDCG@10 medio |
|---|---:|
| `bge-small` fp32, documento truncado a 512 (EDA) | **0.210** (reproducido: iota 0.268, quant 0.151) |
| `bge-base` fp32, documento truncado a 512 | 0.174 (iota 0.208, quant 0.139) |
| `bge-base` fp16, trozos de 254, mejor trozo | 0.177 |
| `bge-base` fp16, trozos de 254, primer trozo | 0.187 |
| BM25 oficial | 0.091 |

`bge-base` rinde igual con y sin chunking, y en fp16 o fp32 (0.174–0.187). **El montaje no es el problema**: en estos dos dominios `bge-base` es, sin más, peor que `bge-small`. Con 31 consultas puede ser ruido. Se confirmará al tener los 13 dominios; si se mantiene, conviene codificar también con `bge-small` (~2.5 veces más rápido) y comparar en la macro.

### 2.7 Denso `bge-base` en los 13 dominios (train)

Corpus completo codificado (1.87 M trozos de 1.17 M documentos, 2.8 GB, ~4.6 h en MPS; corregido el 2026-11-18, antes decía 3.47 M trozos). Macro nDCG@10; Δ y p frente al baseline oficial (todas p < 0.001 salvo `paso`).

**1a** (mejor trozo · primer trozo):

| Variante | nDCG@10 max | nDCG@10 first | R@100 max | R@1000 max |
|---|---:|---:|---:|---:|
| cruda | 0.2150 | 0.2131 | 0.611 | 0.858 |
| limpia | 0.2234 | 0.2216 | 0.617 | 0.872 |
| titulo | 0.2032 | 0.2001 | 0.577 | 0.803 |
| titulo+inicio | 0.2103 | 0.2086 | 0.600 | 0.849 |
| **rrf(limpia,titulo)** | **0.2277** | 0.2251 | 0.641 | 0.863 |
| O1_respuesta (oráculo) | 0.1719 | 0.1658 | 0.531 | 0.805 |
| O1_limpia+respuesta (oráculo) | 0.2343 | 0.2341 | 0.652 | 0.895 |
| O1_rrf(limpia,respuesta) (oráculo) | 0.2014 | 0.1965 | 0.655 | 0.886 |
| O2_normalizada (oráculo) | 0.2165 | 0.2158 | 0.610 | 0.861 |
| O2_limpia+anclas (oráculo) | 0.2214 | 0.2192 | 0.628 | 0.868 |
| O3_resumen (oráculo) | 0.1848 | 0.1863 | 0.545 | 0.786 |
| O3_titulo+resumen (oráculo) | 0.2063 | 0.2078 | 0.584 | 0.833 |
| O3_rrf(limpia,plan) (oráculo) | 0.2179 | 0.2109 | 0.635 | 0.873 |

**1b** (mejor trozo · primer trozo):

| Variante | nDCG@10 max | nDCG@10 first | R@100 max | R@1000 max |
|---|---:|---:|---:|---:|
| oficial (`query + Step`) | 0.2326 | 0.2315 | 0.714 | 0.912 |
| paso | 0.1155 | 0.1147 | 0.447 | 0.711 |
| **limpia+paso** | **0.2449** | 0.2398 | 0.714 | 0.917 |
| titulo+paso | 0.2186 | 0.2128 | 0.706 | 0.901 |
| titulo+paso_limpio | 0.2276 | 0.2201 | 0.727 | 0.903 |
| rrf(paso,limpia) | 0.2066 | 0.2065 | 0.734 | 0.913 |
| rrf(titulo+paso,limpia) | 0.2378 | 0.2360 | 0.751 | 0.926 |

Lecturas:

- **El denso duplica al baseline** y supera con holgura a la mejor variante BM25: 0.228 frente a 0.153 en 1a, y 0.245 frente a 0.166 en 1b. Mejora en 13/13 dominios.
- **Al contrario que BM25, el denso quiere la consulta completa.** El título solo es peor que la consulta limpia, y en 1b el paso aislado se hunde (0.116). Lo que ayuda es quitar el HTML (`limpia`).
- **HyDE con la respuesta gold apenas aporta** (+0.007 sobre `limpia`, como texto añadido) y sola empeora: un HyDE generado por LLM tiene poco margen.
- **Mejor trozo ≥ primer trozo**, pero por muy poco (≤ 0.007). Basta con indexar el principio del documento si hace falta ahorrar.
- En `iota`/`quant`, `bge-base` era peor que `bge-small` (§2.6). Sigue abierto si `bge-small` gana en la macro.

### 2.8 R5 · Híbrido BM25 + denso (RRF, train)

`fuse_runs.py` con RRF (k = 60). Δ y p frente al **mejor denso solo** de cada subtrack, no frente al baseline. `w` es el peso de BM25 (el denso pesa 1).

| Subtrack | Fusión | w BM25 | nDCG@10 | Δ vs denso | p | +/-dom | R@1000 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1a | denso solo `rrf(limpia,titulo)` | — | 0.2277 | — | — | — | 0.863 |
| 1a | BM25 `rrf(limpia,titulo)` + denso `rrf(limpia,titulo)` | 1 | 0.2218 | −0.006 | 0.55 | 6/7 | 0.868 |
| 1a | ídem | 0.5 | 0.2367 | +0.009 | 0.21 | 7/6 | 0.870 |
| 1a | ídem | **0.4** | **0.2390** | +0.011 | 0.09 | 8/5 | 0.872 |
| 1a | BM25 `titulo` + denso `limpia` | 0.5 | 0.2219 | −0.006 | 0.49 | 6/7 | 0.879 |
| 1a | 4 runs: BM25 `titulo`, `limpia` + denso `limpia`, `titulo` | 1 | 0.2321 | +0.004 | 0.63 | 6/7 | 0.872 |
| 1b | denso solo `limpia+paso` | — | 0.2449 | — | — | — | 0.917 |
| 1b | BM25 `titulo+paso_limpio` + denso `limpia+paso` | 1 | 0.2476 | +0.003 | 0.81 | 7/6 | 0.934 |
| 1b | ídem | **0.4** | **0.2611** | +0.016 | 0.03 | 8/5 | 0.937 |
| 1b | ídem | 0.5 | 0.2610 | +0.016 | 0.04 | 8/5 | 0.939 |
| 1b | ídem + denso `rrf(titulo+paso,limpia)` | 1 | 0.2612 | +0.016 | 0.10 | 9/4 | 0.944 |
| 1b | BM25 `oficial` + denso `oficial` | 1 | 0.2180 | −0.027 | 0.007 | 4/9 | 0.901 |

Barrido de w (0.2–0.7): 1a forma una meseta de 0.234–0.239 en w = 0.3–0.6; 1b, de 0.258–0.261 en w = 0.2–0.5.

Lecturas:

- **El híbrido solo ayuda si BM25 pesa menos que el denso.** Con pesos iguales no gana nada: BM25 es mucho peor (0.15 frente a 0.23) y mete ruido arriba del ranking.
- **1b cumple el criterio de aceptación** (Δ ≥ 0.01, p < 0.05, 8/13 dominios) con w = 0.3–0.5. **1a se queda cerca, sin significancia** (+0.011, p = 0.09).
- **Hay que fusionar las mejores variantes de cada retriever.** Fusionar las variantes oficiales empeora.
- **El híbrido sube el recall** (R@1000 de 0.917 a 0.939 en 1b): encuentra más gold entre los 1000 primeros.
- w se eligió en train y la meseta es ancha, así que el riesgo de sobreajuste es bajo. Se fija **w = 0.5** (valor redondo dentro de la meseta) para ambos subtracks hasta la comprobación en dev.

### 2.9 K4 · Década y siglo de las fechas de la consulta (train)

Experimento K4 de [`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md) §6.2. No usa LLM ni `guidance`: todo sale de la consulta con expresiones regulares. La idea: un gold puede hablar del periodo ("the 1830s", "the nineteenth century") sin repetir el año exacto de la consulta.

**Qué se añade.** `reteco.temporal.coarse_forms` saca con reglas las fechas de la consulta (1a: la consulta limpia; 1b: la consulta limpia más la instrucción del paso) y genera sus formas gruesas:

| En la consulta | Se añade |
|---|---|
| "in 1834", "12 March 1834" | "1830s 19th century" |
| "the 1830s" | "19th century" |
| "nineteenth century", "the 1800s" | "19th century" |
| "300 BC" | "3rd century BC" |
| "1914–1918" | "1910s 20th century" |
| "in 2017" | "2010s" (sin "21st century": casi todo el corpus es de este siglo) |

Tampoco se añade la década de los años x00–x09 antes de 2000, porque "the 1800s" se lee como el siglo. No se repite lo que ya está en el texto, y se omiten la actualidad y los rangos de más de 300 años. Antes se corrigió un error del extractor ("twenty first century" sin guion salía como el siglo I; [`RESULTADOS_PERFIL_TEMPORAL.md`](RESULTADOS_PERFIL_TEMPORAL.md) §5).

**Cobertura:**
- 45.2 % de las consultas de 1a y 45.5 % de los pasos de 1b (macro por dominio, 34 %);
- `genealogy` 93–94 %, `politics` 59–60 %, `economics` 49–50 %, `history` 47–49 %, `hsm` 37–41 %, `cardano` 4–6 %, `iota` 0 %;
- mediana de 2 formas por consulta con fecha (p90: 4).

**Tokenización en BM25 (Porter):** "1830s" se reduce a "1830", que choca con el año exacto, y "century" a "centuri".

**Dos formas de usarlo, más un control:**
- **(a) `+k4`**: las formas se añaden al final de cada texto de la variante del brazo.
- **(b) `rrf(...,k4)`**: un texto extra "título. formas" (1b: "título. paso limpio formas") se fusiona con RRF dentro del brazo.
- **Control `k4ctl`**: el mismo texto extra de (b) pero sin las formas. Sirve para separar el efecto de las formas del efecto de añadir un texto más a la fusión.

Las consultas sin fecha no cambian, por construcción. En el denso se ve un ruido de ±0.0005 por el no determinismo de la GPU al codificar.

**Por brazo** (Δ y p frente a la variante base del brazo, todo train):

| Brazo | Base | (a) `+k4` | (b) `rrf(...,k4)` | Control `k4ctl` | (b) − control |
|---|---:|---|---|---|---:|
| BM25 1a `rrf(limpia,titulo)` | 0.1528 | 0.1456 (−0.007, p < 0.001, 2/6) | 0.1546 (+0.002, p 0.26, 9/1) | 0.1549 (+0.002, p 0.15) | −0.000 |
| denso 1a `rrf(limpia,titulo)` | 0.2277 | 0.2255 (−0.002, p 0.35) | 0.2250 (−0.003, p 0.06) | 0.2253 (−0.002, p 0.05) | −0.000 |
| BM25 1b `titulo+paso_limpio` | 0.1660 | 0.1599 (−0.006, p < 0.001, 0/9) | 0.1644 (−0.002, p 0.03) | = base | −0.002 |
| denso 1b `limpia+paso` | 0.2449 | 0.2448 (−0.000, p 0.86) | 0.2501 (+0.005, p 0.07) | 0.2502 (+0.005, p 0.07) | −0.000 |

El control de BM25 1b no hace falta: sin las formas, el texto extra es igual al texto base y la fusión da el mismo ranking.

**Híbrido BM25 + denso** (pesos 0.5 / 1, como en §2.8; Δ frente al híbrido):

| Subtrack | Base | (a) | (b) | Control | (b) − control |
|---|---:|---|---|---|---|
| 1a | 0.2367 (R@1000 0.870) | 0.2340 (−0.003, p 0.28, 4/8; R@1000 0.870) | 0.2348 (−0.002, p 0.35, 7/5; 0.870) | 0.2346 (−0.002, p 0.28, 6/6; 0.872) | +0.000 (p 0.90) |
| 1b | 0.2610 (R@1000 0.939) | 0.2608 (−0.000, p 0.85, 4/4; 0.937) | 0.2652 (+0.004, p 0.04, 7/4; 0.939) | 0.2657 (+0.005, p 0.02, 9/2; 0.940) | −0.000 (p 0.61, 3/8) |

En el control de 1b, BM25 se queda con su variante base y el texto extra va solo al brazo denso.

**En el subconjunto con fecha**, donde está todo el efecto:
- BM25 (a) baja −0.014 (1a) y −0.012 (1b), ambos p < 0.001. Lo que más cae es `genealogy` (−0.044 / −0.046) y `hsm` (−0.054 / −0.017).
- El híbrido con (a) baja −0.010 en 1a (p 0.23, 4/7 dominios) y −0.001 en 1b.
- Ojo: la macro de este subconjunto es ruidosa, porque `cardano` (2 consultas) y `workplace` (3–5) pesan lo mismo que `history` (274–673).

Comando del desglose: `scripts/k4_diag.py --subtrack 1a --coverage --pairs <base> <k4> …`.

Lecturas:

- **K4 no cumple el criterio en ningún brazo ni en el híbrido.** En el híbrido se mueve entre −0.003 y +0.004, y el recall no cambia.
- **Añadir las formas a la consulta (a) empeora BM25 de forma significativa**, sobre todo en `genealogy`, donde casi todas las consultas tienen varias fechas (nacimientos, censos…) y reciben muchas formas. Hay dos causas probables:
  - el stemmer convierte "1830s" en el año "1830", que no está en la consulta;
  - "19th" y "century" son términos genéricos que acercan cualquier documento del siglo.

  En el denso, (a) es neutro: unos tokens más apenas mueven el embedding de una consulta larga.
- **Las pequeñas subidas de (b) no vienen de K4.** En el denso 1b y en el híbrido 1b, (b) sube lo mismo o menos que el control, que lleva el mismo texto extra sin las formas. Las formas, frente al control, quedan en ≤ 0 en todos los brazos y en +0.000 / −0.000 en el híbrido.
- **El extractor con regex funciona; lo que falla es la idea.** El híbrido ya empareja los años exactos (BM25) y el periodo temático (denso); añadir el nombre del periodo no distingue entre candidatos que hablan del mismo tema y de la misma época.
- **Pista aparte, que no es K4:** fusionar con RRF un texto corto "título. paso limpio" con `limpia+paso` en el brazo denso de 1b da +0.005 en el denso (p 0.07) y en el híbrido (p 0.02, 9/2 dominios), aunque aquí solo se aplicó al 45 % de los pasos (los que tienen fecha). Se vio en muestra y queda por debajo de +0.01, así que no se adopta; se apunta como variante de consulta para probar por separado en todos los pasos.

**Decisión:** K4 se cierra. `coarse_forms` se queda en el código, con pruebas, por si sirve más adelante como rasgo.

### 2.10 TT0 · Perfil temporal implícito con n-gramas (resumen)

El detalle está en [`RESULTADOS_PERFIL_TEMPORAL.md`](RESULTADOS_PERFIL_TEMPORAL.md). Es una adaptación del *document focus time* de Jatowt, Au Yeung y Tanaka (CIKM 2013):
- de las oraciones fechadas del corpus se aprende con qué periodos co-ocurre cada unigrama y bigrama, `P(periodo | n-grama)`: un paso de una cadena de Markov n-grama → periodo;
- con eso se da un perfil temporal a consultas y documentos, aunque no tengan fechas;
- se mide si la similitud entre perfiles separa gold de no-gold dentro del top-100 del híbrido BM25 + denso, y qué pasa al fusionarla con él.

En resumen:
- **Señal débil y solo en los dominios históricos.** La AUC es ≈ 0.6 en `history` y `hsm` (también en consultas sin fecha: 0.64 en 565 pasos de `history` 1b) y ≈ 0.5 en `politics`. El propio híbrido separa con 0.77–0.89 en el mismo pool.
- **La fusión con el híbrido sube +0.007 a +0.011 en `history` y `hsm`**, y baja en `politics`. Pero la mejor combinación se elige entre 9 en el propio dominio, cambia entre 1a y 1b y, aplicada en el otro subtrack, queda entre −0.003 y +0.005. En la macro de 13 dominios serían unos +0.001–0.002. No se adopta.
- **El léxico capta palabras ancladas reales** ("henry viii" → 1530s, "qin dynasty" → siglo III a. C.), pero también mucho ruido de páginas repetidas y bibliografía. Los n-gramas más "fechables" son metadatos ("accessed june", "archived november"), igual que en O5.

### 2.11 O4 repetido · Enrutamiento con BM25, denso e híbrido (train)

**Por qué repetirlo.** En §2.4 todas las candidatas eran variantes de BM25 y se parecían demasiado. Ahora hay retrievers cualitativamente distintos: uno léxico (BM25), uno de embeddings (denso) y su híbrido. Si el tipo de razonamiento decidiera qué retriever conviene, aquí se vería.

**Candidatos.**
- **Duro (3 runs):** BM25, denso e híbrido (w 0.5), cada uno con su mejor variante de consulta.
- **Suave (7 runs):** el híbrido con el peso de BM25 en 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 y 1 (el barrido de §2.8). Enrutar aquí es elegir el peso de BM25 en RRF para cada grupo.

**Agrupaciones.**
- `clase` e `intent` reales de `guidance`: el techo de un clasificador perfecto.
- `domain`: el dominio sí se conoce en test, así que este enrutamiento es **realista** y no necesita clasificador.

**Protocolo.**
- Validación cruzada de 5 folds: la regla se elige en 4/5 de las consultas y se aplica al 1/5 restante.
- Los grupos de menos de 10 consultas usan la mejor estrategia única del fold.
- En 1b los folds se agrupan por **consulta madre** ([`scripts/oracle_routing.py`](scripts/oracle_routing.py)), para que los pasos de una misma consulta no queden repartidos entre la elección y la prueba.

**Estrategias únicas** (macro nDCG@10):

| | 1a | 1b |
|---|---:|---:|
| BM25 | 0.1528 | 0.1660 |
| denso | 0.2277 | 0.2449 |
| **híbrido (w 0.5)** | **0.2367** | **0.2610** |
| mejor del barrido suave (w 0.4) | 0.2390 | 0.2611 |

**Enrutamiento** (Δ, p y dominios +/− frente a la mejor estrategia única del conjunto: el híbrido w 0.5 en el duro, w 0.4 en el suave):

| Agrupación | Candidatos | 1a, CV | 1a, en muestra | 1b, CV | 1b, en muestra |
|---|---|---|---:|---|---:|
| clase | duro | 0.2372 (+0.000, p 0.78, 4/6) | 0.2387 | 0.2604 (−0.001, p 0.57, 3/4) | 0.2610 |
| clase | suave | 0.2370 (−0.002, p 0.18, 6/6) | 0.2385 | 0.2615 (+0.000, p 0.88, 5/8) | 0.2639 |
| intención | duro | 0.2344 (−0.002, p 0.49, 5/7) | 0.2372 | 0.2561 (−0.005, p 0.13, 6/7) | 0.2604 |
| intención | suave | 0.2342 (−0.005, p 0.04, 6/7) | 0.2399 | 0.2614 (+0.000, p 0.91, 8/5) | 0.2627 |
| dominio | duro | 0.2417 (+0.005, p 0.17, 2/3) | 0.2488 | 0.2577 (−0.003, p 0.43, 3/4) | 0.2719 |
| dominio | suave | 0.2397 (+0.001, p 0.85, 6/6) | 0.2467 | 0.2600 (−0.001, p 0.74, 6/6) | 0.2698 |
| oráculo por consulta | duro / suave | 0.3030 / 0.2734 | | 0.3436 / 0.3039 | |

**Por clase** (en muestra):
- **1b:** el híbrido es el mejor candidato en las 7 clases.
- **1a:** el híbrido gana en EAL, OEC, TPC y TCP; el denso solo le gana en EVA (+0.013), MAP (+0.010) y `other` (+0.011), clases de 68–84 consultas.
- **Intenciones (1a):** el denso gana en `duration` (+0.017), `ongoing_status` (+0.001) y `other` (+0.006); el híbrido, en las demás.

**Por dominio** (en muestra), el patrón más claro es que algunos dominios prefieren el denso solo, sin BM25, en los dos subtracks:

| | `travel` | `monero` | `bitcoin` | `quant` |
|---|---:|---:|---:|---:|
| 1a (denso − híbrido) | +0.071 | +0.032 | +0.004 | +0.013 |
| 1b (denso − híbrido) | +0.029 | +0.057 | +0.018 | +0.002 |

En el suave, esos mismos dominios eligen el peso más bajo de BM25 (w 0.2), y `hsm` y `politics` el más alto (w 0.7–1). Pero 1a y 1b comparten consultas, así que no son evidencia independiente, y son dominios de 24–133 consultas.

Lecturas:

- **Ni con etiquetas perfectas el enrutamiento por clase o intención supera al híbrido.**
  - Con validación cruzada queda entre −0.005 y +0.000.
  - En muestra queda a ≤ +0.003.
  - Las clases donde el denso gana en 1a son pequeñas, y en 1b el híbrido gana en todas: la fusión ya absorbe las diferencias entre retrievers.
- **Elegir el peso de BM25 por grupo (enrutamiento suave) tampoco ayuda.** El barrido de §2.8 es una meseta: ningún grupo tiene un óptimo claramente distinto.
- **El enrutamiento por dominio parece bueno en muestra (+0.010 a +0.012), pero es sobreajuste.**
  - Con validación cruzada queda en +0.005 en 1a (p 0.17) y en −0.003 en 1b.
  - Lo único estable es que `travel` y `monero` prefieren el denso solo, pero no basta para mover la macro.
- **El oráculo por consulta es alto** (0.30 / 0.34 con los candidatos duros, +0.07 / +0.08 sobre el híbrido).
  - Hay mucha variación entre consultas, pero ni la clase, ni la intención, ni el dominio la explican.
  - Solo se podría aprovechar con un modelo que mire cada consulta y cada documento, no con un enrutador por tipo.

**Decisión:**
- **Fase 3 (clasificador) cerrada definitivamente.**
- Los pesos por dominio no se adoptan.
- El híbrido BM25 + denso con w 0.5 sigue siendo la primera etapa para todas las consultas.

Salidas: `outputs/exp/oracles/O4_routing_<subtrack>_<grupo>_{bm25,duro,suave}.csv` (tablas por grupo).

---

## 3. Conclusiones

1. **Mejoras conseguidas sin modelos**, solo con expresiones regulares y la estructura de la consulta:
   - 1a: `rrf(limpia,titulo)`, **0.153** (+42 % relativo);
   - 1b: `titulo+paso_limpio`, **0.166** (+62 %).
2. **Con BM25, añadir texto empeora y quitar ruido mejora.** HyDE, etiquetas temporales y razonamientos largos no ayudan a un retriever léxico.
3. **Clasificar el tipo de razonamiento no aporta**: ni con etiquetas perfectas mejora el ranking, ni eligiendo entre variantes de BM25 (§2.4) ni entre BM25, denso e híbrido o sus pesos (§2.11). **Fase 3 cerrada.** Tampoco sirve elegir el retriever por dominio. La variación entre consultas es grande (oráculo por consulta 0.30 / 0.34), pero no la explica el tipo de consulta.
4. **No hay un vocabulario de documento por clase** (O5). Los gold se distinguen por ser contenido narrativo y no bibliografía, igual en todas las clases.
5. **Los embeddings son la base** (§2.7). Con `bge-base`, 1a `rrf(limpia,titulo)` da 0.228 y 1b `limpia+paso` da 0.245, unas 2.1–2.4 veces el baseline oficial. HyDE con respuesta perfecta solo añade ~0.007.
6. **La mejor configuración del proyecto es el híbrido BM25 + denso con BM25 a medio peso** (w = 0.5, §2.8): **0.2367 en 1a y 0.2610 en 1b**, 2.2–2.5 veces el baseline, con R@1000 de 0.870 y 0.939. En 1b la mejora sobre el denso solo es significativa. Falta confirmarlo en dev, al final.
7. **Añadir la década y el siglo de las fechas de la consulta (K4) no ayuda** (§2.9). Mete ruido en BM25 y es neutro en el denso y en el híbrido. Las pequeñas subidas que se vieron vienen de añadir un texto más a la fusión, no de las formas temporales.
8. **El periodo implícito de un texto se puede inferir de sus n-gramas, pero apenas separa gold de no-gold mejor que el híbrido** (§2.10; [`RESULTADOS_PERFIL_TEMPORAL.md`](RESULTADOS_PERFIL_TEMPORAL.md)). En `history` y `hsm` la fusión gana algo (+0.007 a +0.011, elegida en muestra), pero no es estable ni mueve la macro.
9. **Lectura de conjunto:** tras K4 y TT0, dos maneras distintas de usar el tiempo de forma explícita (fechas con regex; periodo implícito con n-gramas y una cadena de Markov), la conclusión es la misma: el híbrido ya captura casi toda la coincidencia temporal que se puede sacar sin entender la consulta. Los candidatos del top-100 ya hablan del mismo tema y de la misma época; lo que distingue al gold es otra cosa (que responda a la pregunta, que sea narrativo y no bibliográfico).

## 4. Pendiente / siguientes pasos

1. ✅ Codificación `bge-base` completa y variantes 1a/1b evaluadas, con mejor trozo y con primer trozo (§2.7).
2. ✅ R5: híbrido BM25 + denso (§2.8).
3. ✅ O4 repetido con BM25, denso e híbrido (§2.11): ni por clase, ni por intención, ni por dominio. Fase 3 cerrada.
4. ✅ K4 (§2.9): no ayuda, se cierra.
5. ✅ TT0, perfil temporal implícito (§2.10): señal débil; queda como posible rasgo.
6. Decidir si codificar también con `bge-small` (~2 h) para compararlo en la macro.
7. Probar el rasgo "bibliografía/boilerplate" (O5) como penalización suave del score.
8. Probar por separado, en todos los pasos de 1b, el texto extra "título. paso limpio" fusionado con RRF en el brazo denso (+0.005 en el 45 % de los pasos, §2.9).
9. Si se retoma TT0, elegir la fusión con validación cruzada en los dominios históricos y medirla con test pareado.
10. Comprobación única en dev del híbrido (y de lo que pase el criterio en train), al final.
