# Resultados de los experimentos — Fases 0–2 y TG0 (estado al 2026-11-11)

Resumen de lo implementado de [`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md) y [`PLAN_TGRAG.md`](PLAN_TGRAG.md), y de los resultados obtenidos hasta ahora. Todas las cifras son **macro nDCG@10 en train** (13 dominios), salvo que se indique otra cosa. Dev no se ha tocado.

**Estado:**

| Fase | Estado |
|---|---|
| 0 · Infraestructura | ✅ terminada; reproduce exactamente el baseline oficial |
| 1 · Retrievers de referencia | ✅ BM25, denso `bge-base` (13 dominios) e híbrido R5 |
| 2 · Oráculos | ✅ O1–O3 con BM25 y con el denso; O5. ⏳ Falta repetir O4 con candidatos densos |
| 3–5 · Clasificador, keywords, LLM | sin empezar |
| TG-RAG | ✅ revisión del extractor ([`REVISION_EXTRACCION_TGRAG.md`](REVISION_EXTRACCION_TGRAG.md)), TG0 (§2.9) y TG1–TG6 + TG8 (§2.10). ❌ **Rama cerrada:** ninguna variante mejora el híbrido |

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

### 2.9 TG0 · ¿Puede ayudar el filtro temporal de TG-RAG? (train, 2026-11-11)

Diagnóstico de cobertura de [`PLAN_TGRAG.md`](PLAN_TGRAG.md) §4, **sin ranking**. Corrió en `myserver` (`~/tg0_run.sh`, 16 min):

- **Pools:** top-100 del híbrido de §2.8 (w = 0.5) de cada subtrack (`tg_pool.py --depth 100 --name p100`). 1a: 1,211 consultas, ~89 k documentos únicos; 1b: 2,762 consultas, ~95 k.
- **Hechos** con reglas en dos variantes: `rules` y `rules-nocit-nometa` (sin años de cita ni fechas de la página; ver [`REVISION_EXTRACCION_TGRAG.md`](REVISION_EXTRACCION_TGRAG.md) §3.3).
- **`T_q`** con reglas y con oráculo (`guidance`).
- **`tg_diagnose.py`** para las 8 combinaciones. Salidas en el servidor: `outputs/exp/tgrag/diag/tg0_*.csv` y `outputs/exp/logs/tg0.log`.

Métricas (entre las consultas con `T_q` no vacío): `gold_en_Tq` / `nogold_en_Tq` es la fracción de documentos gold / no gold del pool con algún hecho cuyo τ cae en `T_q`. El **lift** es su cociente: ≈ 1 significa que el filtro no separa gold de no-gold. Usa qrels de train: es un análisis, no un sistema.

**Resumen** (macro, emparejamiento `contained`):

| Subtrack | Extractor | `T_q` | Con `T_q` | Recall pool | Doc. fechados | Gold en `T_q` | No-gold en `T_q` | Cociente de medias | Lift macro |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1a | `rules` | reglas | 0.359 | 0.657 | 0.422 | 0.212 | 0.165 | **1.28** | 1.076 |
| 1a | `rules` | oráculo | 0.389 | 0.657 | 0.422 | 0.234 | 0.169 | 1.38 | 1.152 |
| 1a | `rules-nocit-nometa` | reglas | 0.359 | 0.657 | 0.406 | 0.206 | 0.157 | 1.31 | 1.092 |
| 1a | `rules-nocit-nometa` | oráculo | 0.389 | 0.657 | 0.406 | 0.225 | 0.162 | 1.39 | 1.151 |
| 1b | `rules` | reglas | 0.363 | 0.759 | 0.429 | 0.244 | 0.157 | **1.55** | 1.217 |
| 1b | `rules` | oráculo | 0.399 | 0.759 | 0.429 | 0.259 | 0.170 | 1.52 | 1.202 |
| 1b | `rules-nocit-nometa` | reglas | 0.363 | 0.759 | 0.413 | 0.237 | 0.151 | 1.57 | 1.229 |
| 1b | `rules-nocit-nometa` | oráculo | 0.399 | 0.759 | 0.413 | 0.250 | 0.163 | 1.53 | 1.188 |

> **Ojo con el "lift macro"** que imprime `tg_diagnose.py`: es la media de los cocientes por dominio y cuenta como 0 a `quant` y `workplace` (y a `cardano` en 1b), que tienen 2–10 consultas con `T_q` y ningún gold dentro. Eso lo hunde. El **cociente de las medias macro** (`gold_en_Tq` / `nogold_en_Tq`) describe mejor la señal y es el que se usa en las lecturas.

**Por dominio** (`rules`, `T_q` reglas, `contained`):

| Dominio | Consultas con `T_q` 1a / 1b | Gold / no-gold en `T_q` 1a | Lift 1a | Gold / no-gold en `T_q` 1b | Lift 1b |
|---|---:|---:|---:|---:|---:|
| history | 299 / 754 | 0.383 / 0.261 | 1.47 | 0.436 / 0.273 | 1.60 |
| politics | 64 / 160 | 0.427 / 0.359 | 1.19 | 0.502 / 0.327 | 1.54 |
| genealogy | 75 / 147 | 0.392 / 0.345 | 1.14 | 0.447 / 0.285 | 1.57 |
| hsm | 60 / 142 | 0.528 / 0.319 | 1.66 | 0.551 / 0.319 | 1.73 |
| economics | 29 / 64 | 0.366 / 0.230 | 1.59 | 0.452 / 0.269 | 1.68 |
| travel | 17 / 33 | 0.107 / 0.112 | 0.95 | 0.233 / 0.128 | 1.82 |
| monero | 14 / 29 | 0.080 / 0.070 | 1.15 | 0.130 / 0.063 | 2.07 |
| bitcoin | 12 / 26 | 0.250 / 0.147 | 1.71 | 0.304 / 0.166 | 1.83 |
| law | 8 / 16 | 0.222 / 0.225 | 0.99 | 0.111 / 0.145 | 0.77 |
| quant | 7 / 10 | 0 / 0.052 | 0 | 0 / 0.039 | 0 |
| workplace | 3 / 5 | 0 / 0.024 | 0 | 0 / 0.026 | 0 |
| cardano | 2 / 2 | 0 / 0 | — | 0 / 0.005 | 0 |
| iota | 0 / 0 | — | — | — | — |

Por modo de emparejamiento (lift macro, `rules`, `T_q` reglas): 1a `contained` 1.08 · `overlap` 0.88 · `soft≥0.5` 0.99; 1b 1.22 · 1.08 · 1.00.

Lecturas:

- **Se cumple la condición para seguir.** El plan decía parar si los gold no se concentran en `T_q`. Sí se concentran: un gold tiene 1.3 veces (1a) y 1.55 veces (1b) más probabilidad que un no-gold de tener un hecho en `T_q`.
- **La señal está en los dominios históricos**, como predecía H1. En los cinco con más consultas fechadas (`history`, `politics`, `genealogy`, `hsm`, `economics`) el lift es 1.1–1.7 en 1a y 1.5–1.7 en 1b. En 1b la separación es mayor en casi todos los dominios.
- **Como filtro duro perdería más de la mitad de los gold.** Incluso en `history` solo el 38 % (1a) y el 44 % (1b) de los gold tienen un hecho en `T_q`; en macro, el 21 % y el 24 %. Coincide con el EDA 6.3 (el 55 % de los pares consulta–gold no comparte año). En TG1 el score temporal tiene que entrar **fusionado** con el híbrido, y el desglose con y sin `T_q` es obligatorio.
- **Solo el 36 % de las consultas tiene `T_q`** con reglas (39 % con oráculo). En el 64 % restante TG-RAG solo puede aportar por el grafo de entidades.
- **El `T_q` oráculo sube la cobertura, no la separación.** `history` pasa de 53 % a 71 % de consultas con `T_q`, pero el cociente casi no cambia (1a 1.28 → 1.38; 1b 1.55 → 1.52). Un LLM para `T_q` (TG3) ayudaría sobre todo en cobertura, y sobre todo en `history`.
- **`contained` es el único emparejamiento que separa.** Con `overlap` y `soft` casi cualquier documento fechado casa con `T_q` y gold y no-gold se igualan. TG4 tiene poco recorrido.
- **Quitar citas y metadatos es neutro**: el lift apenas se mueve (1a 1.076 → 1.092, 1b 1.217 → 1.229) y los gold en `T_q` bajan un poco (0.212 → 0.206). Se sigue solo con `rules`.
- **Dominios técnicos sin base temporal:** `iota` no tiene consultas con `T_q`; `cardano`, `workplace` y `quant` tienen 2–7 en 1a y ningún gold en `T_q`. Allí hay que usar el *fallback* sin filtro.
- **Techo del reranker:** el pool top-100 contiene el 65.7 % de los gold en 1a y el 75.9 % en 1b.

### 2.10 TG1–TG6 y TG8 · TG-RAG como reranker (train, 2026-11-11)

Experimentos de [`PLAN_TGRAG.md`](PLAN_TGRAG.md) §4 que no necesitan LLM. Corrieron en `myserver` (`~/tg1_run.sh` y `~/tg5_run.sh`, ~25 min en total con una RTX 3070).

**Configuración común:**
- pool `p100` de §2.9 y hechos `rules`;
- encoder `bge-small` para aristas y consultas;
- `e_q`: `limpia` en 1a y `limpia+paso` en 1b. En 1b se cambió el `titulo+paso_limpio` que fijaba el plan, porque era la mejor variante para BM25 y el denso prefiere la consulta completa (§2.7);
- base: el híbrido de §2.8 con w = 0.5 (1a 0.2367, 1b 0.2610). Δ, p y dominios son siempre **frente a esa base**, no frente al baseline oficial.
- Cada configuración da un run TG puro (solo puntúa los documentos con alguna arista en `E_q`) y su fusión RRF con la base, con peso `w` para TG.

Se usan los nombres de los presets del plan:

| Preset | Qué hace |
|---|---|
| `full` | el Algoritmo 1 completo del paper |
| `no_ppr` | sin la propagación por el grafo (PPR) |
| `no_temporal` | sin el filtro temporal `T_q` |
| `no_temporal_ppr` | sin filtro temporal ni PPR: solo el coseno de las aristas |
| `static` | sin τ en las aristas (grafo estático) |

**TG1 + TG2** (`T_q` reglas, `contained`, K = 20):

| Preset | 1a TG puro | 1a +rrf w=0.5 (Δ, p, +/−dom) | 1a +rrf w=1 | 1b TG puro | 1b +rrf w=0.5 (Δ, p, +/−dom) | 1b +rrf w=1 |
|---|---:|---|---:|---:|---|---:|
| `full` | 0.126 | 0.2210 (−0.016, 0.03, 2/11) | 0.2114 | 0.123 | 0.2274 (−0.034, <0.001, 1/12) | 0.2140 |
| `no_ppr` | 0.128 | 0.2206 (−0.016, 0.02, 2/11) | 0.2103 | 0.125 | 0.2285 (−0.033, <0.001, 1/12) | 0.2177 |
| `no_temporal` | 0.135 | 0.2214 (−0.015, 0.03, 2/11) | 0.2109 | 0.134 | 0.2293 (−0.032, <0.001, 1/12) | 0.2156 |
| `no_temporal_ppr` | 0.137 | 0.2210 (−0.016, 0.03, 2/11) | 0.2101 | 0.136 | 0.2306 (−0.030, <0.001, 1/12) | 0.2195 |
| `static` | 0.144 | 0.2156 (−0.021, 0.003, 1/12) | 0.2031 | 0.146 | 0.2262 (−0.035, <0.001, 1/12) | 0.2134 |

El run TG puro tiene R@100 de 0.18 (1a) y 0.22 (1b), frente a 0.66 y 0.76 del pool.

**TG3 (`T_q` oráculo), TG4 (emparejamiento) y extractor sin citas** (`full`, +rrf w = 0.5):

| Variante | 1a | 1b |
|---|---:|---:|
| `T_q` reglas, `contained` (TG1) | 0.2210 | 0.2274 |
| `T_q` **oráculo**, `contained` | 0.2216 | 0.2283 |
| `T_q` reglas, `overlap` / `soft` | 0.2208 / 0.2201 | — |
| `T_q` oráculo, `overlap` / `soft` | 0.2218 / 0.2200 | — |
| hechos `rules-nocit-nometa` | 0.2204 | — |

**TG5 (cobertura) y TG6 (peso de fusión)**, 1a, `full`, `T_q` reglas:

| Configuración | TG puro (R@100) | +rrf w=0.1 | +rrf w=0.25 | +rrf w=0.5 |
|---|---:|---:|---:|---:|
| K = 20, `subgraph` (TG1) | 0.126 (0.18) | **0.2413** (+0.005, p 0.25, 8/5) | 0.2358 (−0.001, p 0.88, 3/10) | 0.2210 |
| K = 50 | — | — | 0.2300 | 0.2127 |
| K = 200 | 0.086 (0.38) | — | 0.2234 | 0.2051 |
| K = 50, `graph_scope = pool` | 0.035 (0.53) | — | 0.2080 | 0.1827 |
| K = 200, `graph_scope = pool` | 0.034 (0.51) | — | 0.2053 | 0.1794 |

`no_temporal` con los mismos K y `graph_scope` da lo mismo, ±0.004.

**Por dominio** (Δ frente a la base; `full`, `T_q` reglas, +rrf w = 0.5):

| | bitcoin | cardano | economics | genealogy | history | hsm | iota | law | monero | politics | quant | travel | workplace |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1a | −0.018 | +0.014 | −0.025 | −0.038 | −0.008 | −0.021 | −0.020 | −0.032 | −0.014 | −0.039 | −0.007 | −0.022 | +0.025 |
| 1b | −0.021 | −0.003 | −0.036 | −0.046 | −0.020 | −0.026 | −0.062 | −0.047 | −0.071 | −0.049 | −0.020 | −0.051 | +0.014 |

**Desgloses obligatorios** (Δ macro frente a la base, +rrf w = 0.5):

| Grupo | 1a `full` | 1a `no_temporal` | 1b `full` | 1b `no_temporal` |
|---|---:|---:|---:|---:|
| consultas **con** `T_q` (590 / 1,388) | −0.011 | −0.008 | −0.011 | **−0.003** |
| consultas **sin** `T_q` (621 / 1,374) | −0.017 | −0.017 | −0.037 | −0.037 |
| clase EAL (435 / 974) | −0.030 | −0.028 | −0.026 | −0.025 |
| clase TPC (256 / 604) | +0.001 | −0.007 | −0.010 | −0.019 |
| intención `period_definition` (244 / 597) | −0.062 | −0.075 | −0.091 | −0.100 |
| intención `timeline` (243 / 629) | −0.029 | −0.022 | −0.063 | −0.048 |

Los únicos grupos positivos son pequeños y salen igual sin el componente temporal: provenance (MAP) +0.048, n = 68; verificación (EVA) +0.036, n = 80; ambos en 1a.

Salidas en el servidor: runs en `outputs/exp/runs/tgrag-*`, logs `outputs/exp/logs/tg1.log`, `tg5.log`, `tg_perdomain.log` y `tg1_desgloses.log`. En el Mac están copiados `resultados.csv`, `tgrag/{diag,pools,qtime}` y los logs. Los hechos (1.6 GB) y los embeddings de aristas (1.9 GB) solo están en el servidor.

Lecturas:

- **Ninguna variante mejora el híbrido.** Con w = 0.5 todas empeoran de forma significativa en 1a (−0.015 a −0.021) y en 1b (−0.030 a −0.035), en 11–12 de 13 dominios. Al bajar el peso solo se acercan a la base. w = 0.1 da +0.005 (p = 0.25), por debajo del criterio de aceptación (Δ ≥ 0.01, p < 0.05), y además es el mejor de 4 pesos elegido en muestra.
- **El score TG es un retriever mucho más débil que la base:** 0.13 frente a 0.24 en 1a. Incluso `no_temporal_ppr`, que es solo el coseno de `bge-small` entre la consulta y las oraciones-arista, da 0.137. Fusionarlo trae documentos peores al top-10.
- **No se reproduce la tabla VII del paper (H2 rechazada).** El componente temporal no aporta:
  - `full` ≈ `no_temporal` en la macro;
  - en las consultas con `T_q` incluso empeora (1b: −0.011 frente a −0.003);
  - el PPR tampoco aporta (`no_ppr` ≈ `full`).

  El lift de 1.3–1.55 de TG0 no basta. Hipótesis, no comprobada: el híbrido ya premia los documentos con las mismas fechas que la consulta, porque BM25 empareja los años como términos.
- **El `T_q` oráculo no cambia nada** (0.2216 frente a 0.2210 en 1a). Por la regla del plan ("si TG3-oráculo no muestra recorrido, se cierra la rama sin gastar LLM") **no se hacen TG3-LLM ni TG7** (extracción con LLM).
- **Más cobertura empeora** (TG5). Con K = 200 o con el grafo de todo el pool se puntúan más documentos (R@100 0.38–0.53), pero con peor orden: el run puro baja a 0.03–0.09.
- **Ni la clase ni la intención temporal rescatan el método** (H1 rechazada). EAL baja; TPC se queda en +0.001. Lo que más se hunde son `period_definition` y `timeline`, las intenciones que más deberían beneficiarse de un filtro temporal.

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
7. **El filtro temporal de TG-RAG tiene recorrido, pero solo como señal y solo en los dominios históricos** (§2.9). Los gold caen en `T_q` 1.3 veces (1a) y 1.55 veces (1b) más que los no-gold, pero menos de la mitad de los gold tienen un hecho en `T_q`. Un filtro duro no sirve.
8. **TG-RAG como reranker no funciona en TEMPO** (§2.10). Todas las variantes sin LLM (ablaciones, `T_q` oráculo, emparejamientos, más cobertura, pesos de fusión) quedan por debajo del híbrido o empatan con él. Ni el componente temporal ni el PPR aportan. **Rama cerrada** sin gastar LLM. La coincidencia de periodo (§2.9) puede servir todavía como **rasgo** de un reranker aprendido, no como método propio.

## 4. Pendiente / siguientes pasos

1. ✅ Codificación `bge-base` completa y variantes 1a/1b evaluadas, con mejor trozo y con primer trozo (§2.7).
2. ✅ R5: híbrido BM25 + denso (§2.8).
3. ✅ TG0 (§2.9) y ✅ TG1–TG6 + TG8 (§2.10): rama TG-RAG cerrada. Si se retoma, solo como rasgo "hecho con τ en `T_q`" para el reranker cross-encoder, no como reranker propio.
4. Decidir si codificar también con `bge-small` (~2 h) para compararlo en la macro.
5. Repetir O4 con candidatos léxicos + densos + oráculos de reescritura.
6. Fase 5 con LLM local (`mlx-lm` + Qwen), solo para lo que el denso muestre que tiene recorrido: condensación (Q1) y HyDE (Q3) primero.
7. Probar el rasgo "bibliografía/boilerplate" (O5) como penalización suave antes del reranker.
