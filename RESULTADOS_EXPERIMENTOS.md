# Resultados de los experimentos — Fases 0–2 (estado al 2026-11-07)

Resumen de lo implementado de [`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md) y de los resultados obtenidos hasta ahora. Todas las cifras son **macro nDCG@10 en train** (13 dominios), salvo que se indique otra cosa. Dev no se ha tocado.

**Estado:**

| Fase | Estado |
|---|---|
| 0 · Infraestructura | ✅ terminada; reproduce exactamente el baseline oficial |
| 1 · Retrievers de referencia | ⏳ BM25 listo; denso (`bge-base`) codificando: 4 de 13 dominios |
| 2 · Oráculos | ✅ O1–O3 y O5 con BM25; O4 con variantes léxicas. ⏳ Falta repetir O1–O4 con el denso |
| 3–5 · Clasificador, keywords, LLM | sin empezar |

---

## 1. Qué se creó

### Módulos (`src/reteco/`)

| Archivo | Contenido |
|---|---|
| [`data.py`](src/reteco/data.py) | rutas, cargadores (consultas, pasos, qrels, guidance), limpieza HTML, normalización de etiquetas (`clase`: 6 clases + `other`; `intent`: 6 intenciones + `other`) |
| [`bm25.py`](src/reteco/bm25.py) | `OfficialStyleBM25` del notebook, reutilizando los índices de `outputs/eda/bm25/` |
| [`dense.py`](src/reteco/dense.py) | encoders (`bge-small`, `bge-base`, `e5-base`) en fp16 sobre MPS; chunking de 254 tokens, solape de 30 y máximo 4 trozos por documento; agregación `max` (mejor trozo) o `first` (primer trozo) |
| [`fusion.py`](src/reteco/fusion.py) | `top_k` con el desempate del oficial y RRF (también por consulta con varios textos) |
| [`queries.py`](src/reteco/queries.py) | registro de variantes de consulta de 1a y 1b, incluidas las de oráculo y `strip_step` (quita el verbo imperativo y las palabras meta de los pasos) |
| [`evaluate.py`](src/reteco/evaluate.py) | `pytrec_eval`, macro por dominio, test pareado de aleatorización (sobre la macro), desglose por clase e intención, registro en `outputs/exp/resultados.csv` |

`bm25` (JVM) y `dense` (torch) se importan solo en procesos distintos, para evitar la caída del kernel que se vio en el EDA.

### Scripts (`scripts/`)

| Script | Uso |
|---|---|
| [`run_experiment.py`](scripts/run_experiment.py) | `--subtrack 1a --retriever bm25 --variants all`: corre, evalúa y compara con el baseline oficial; reutiliza los runs ya calculados |
| [`encode_corpus.py`](scripts/encode_corpus.py) | `--model bge-base`: codifica los corpus por trozos; se puede interrumpir y relanzar |
| [`fuse_runs.py`](scripts/fuse_runs.py) | RRF de runs ya calculados (p. ej. el híbrido BM25 + denso, R5) |
| [`oracle_routing.py`](scripts/oracle_routing.py) | O4: mejor estrategia única frente al enrutamiento por clase o intención (con CV) y frente al oráculo por consulta |
| [`oracle_class_terms.py`](scripts/oracle_class_terms.py) | O5: log-odds de términos de los gold frente a los hard negatives, por clase |

### Salidas (`outputs/exp/`)

- `runs/<retriever>/<variante>/<dominio>__train.pkl`: rankings top-1000.
- `resultados.csv`: una fila por experimento (nDCG@10, R@10/100/1000, Δ, p, dominios que mejoran / empeoran).
- `oracles/`: tablas de O4 y O5.
- `dense/bge-base/`: embeddings por dominio.
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

**Lectura:** para BM25, el techo de "reescritura con LLM" que dan los textos de `guidance` apenas supera la heurística del título. Lo que importa es **quitar ruido** a la consulta, no añadir texto. Estas conclusiones valen para BM25; el paper indica que los encoders densos, sobre todo los que siguen instrucciones, reaccionan distinto. **Falta repetir O1–O3 con el denso.**

### 2.4 O4 · ¿Sirve clasificar la consulta para elegir la estrategia?

Candidatos: las variantes realistas de BM25 de §2.2. Enrutamiento con etiquetas **reales** de guidance y CV de 5 folds.

| | 1a por clase | 1a por intención | 1b por clase |
|---|---:|---:|---:|
| Mejor estrategia única | 0.1528 | 0.1528 | 0.1660 |
| Enrutamiento (CV) | 0.1436 (Δ −0.009, p 0.14) | 0.1472 (Δ −0.006, p 0.43) | 0.1667 (Δ +0.001, p 0.42) |
| Oráculo por consulta | 0.2201 | 0.2201 | 0.2413 |

- **Ni con etiquetas perfectas el enrutamiento por clase mejora.** La misma variante gana en casi todas las clases (el título en 1a, `titulo+paso_limpio` en 1b, en las 7 clases).
- El oráculo por consulta (0.22) muestra que **sí hay variación entre consultas, pero el tipo de razonamiento no la explica**.
- **Matiz:** las candidatas son todas léxicas y parecidas entre sí. Hay que repetir O4 cuando haya estrategias cualitativamente distintas (denso, HyDE, descomposición por periodo). De momento, la Fase 3 (clasificador) **no está justificada**.

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

**Decisión:** descartar K1/K2 (expansión con keywords por clase). La señal "bibliografía / boilerplate" se apunta como **rasgo de calidad del documento** para el reranker. Es un rasgo de contenido, no del formato del ID.

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

Corpus completo codificado (3.47 M trozos, 2.8 GB, ~4.6 h en MPS). Macro nDCG@10; Δ y p frente al baseline oficial (todas p < 0.001 salvo `paso`).

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
- **El híbrido sube el recall** (R@1000 de 0.917 a 0.939 en 1b), lo que importa para la lista de candidatos del reranker.
- w se eligió en train y la meseta es ancha, así que el riesgo de sobreajuste es bajo. Se fija **w = 0.5** (valor redondo dentro de la meseta) para ambos subtracks hasta la comprobación en dev.

---

## 3. Conclusiones hasta ahora

1. **Mejoras ya conseguidas sin modelos** (candidatas a línea base nueva):
   - 1a: `rrf(limpia,titulo)`, **0.153** (+42 % relativo);
   - 1b: `titulo+paso_limpio`, **0.166** (+62 %).
2. **Con BM25, añadir texto empeora y quitar ruido mejora.** HyDE, etiquetas temporales y razonamientos largos no ayudan a un retriever léxico.
3. **Clasificar el tipo de razonamiento no aporta**, de momento: ni con etiquetas perfectas mejora el ranking.
4. **No hay un vocabulario de documento por clase.** Los gold se distinguen por ser contenido narrativo y no bibliografía, igual en todas las clases.
5. **El retriever denso es la nueva base** (§2.7). Con `bge-base`, 1a `rrf(limpia,titulo)` da 0.228 y 1b `limpia+paso` da 0.245, unas 2.1–2.4 veces el baseline oficial. HyDE con respuesta perfecta solo añade ~0.007.
6. **El híbrido con BM25 a medio peso (w = 0.5) es la mejor configuración hasta ahora** (§2.8): 0.237 en 1a y 0.261 en 1b. En 1b la mejora sobre el denso solo es significativa.

## 4. Pendiente / siguientes pasos

1. ✅ Codificación `bge-base` completa y variantes 1a/1b evaluadas, con mejor trozo y con primer trozo (§2.7).
2. ✅ R5: híbrido BM25 + denso (§2.8).
3. Decidir si codificar también con `bge-small` (~2 h) para compararlo en la macro.
4. Repetir O4 con candidatos léxicos + densos + oráculos de reescritura.
5. Fase 5 con LLM local (`mlx-lm` + Qwen), solo para lo que el denso muestre que tiene recorrido: condensación (Q1) y HyDE (Q3) primero.
6. Probar el rasgo "bibliografía/boilerplate" (O5) como penalización suave antes del reranker.
