# Resultados de TG-RAG (Temporal GraphRAG) — estado al 2026-11-11

Resultados de lo implementado de [`PLAN_TGRAG.md`](PLAN_TGRAG.md), la adaptación a TEMPO de Han et al., *RAG Meets Temporal Graphs* ([`papers/2510.13590v1.pdf`](papers/2510.13590v1.pdf)). La revisión del extractor con reglas, previa a TG0, está en [`REVISION_EXTRACCION_TGRAG.md`](REVISION_EXTRACCION_TGRAG.md). Los demás experimentos (BM25, denso, híbrido, oráculos) están en [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md).

Las reglas son las de siempre: todo se ajusta en train, dev no se ha tocado, y `guidance` solo aparece en las variantes marcadas como oráculo.

**Estado:**

| Experimento | Estado |
|---|---|
| Revisión del extractor con reglas | ✅ ([`REVISION_EXTRACCION_TGRAG.md`](REVISION_EXTRACCION_TGRAG.md)) |
| TG0 · diagnóstico de cobertura | ✅ §1: los gold se concentran en `T_q` → seguir |
| TG1–TG6 y TG8 · reranker sin LLM | ✅ §2: ninguna variante mejora el híbrido |
| TG3-LLM y TG7 · `T_q` y extracción con LLM | ❌ no se hacen: **rama cerrada** (TG3-oráculo sin recorrido) |

---

## 1. TG0 · ¿Puede ayudar el filtro temporal? (train, 2026-11-11)

Diagnóstico de cobertura de [`PLAN_TGRAG.md`](PLAN_TGRAG.md) §4, **sin ranking**. Corrió en `myserver` (`~/tg0_run.sh`, 16 min):

- **Pools:** top-100 del híbrido de [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) §2.8 (w = 0.5) de cada subtrack (`tg_pool.py --depth 100 --name p100`). 1a: 1,211 consultas, ~89 k documentos únicos; 1b: 2,762 consultas, ~95 k.
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

## 2. TG1–TG6 y TG8 · TG-RAG como reranker (train, 2026-11-11)

Experimentos de [`PLAN_TGRAG.md`](PLAN_TGRAG.md) §4 que no necesitan LLM. Corrieron en `myserver` (`~/tg1_run.sh` y `~/tg5_run.sh`, ~25 min en total con una RTX 3070).

**Configuración común:**
- pool `p100` de §1 y hechos `rules`;
- encoder `bge-small` para aristas y consultas;
- `e_q`: `limpia` en 1a y `limpia+paso` en 1b. En 1b se cambió el `titulo+paso_limpio` que fijaba el plan, porque era la mejor variante para BM25 y el denso prefiere la consulta completa ([`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) §2.7);
- base: el híbrido de [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) §2.8 con w = 0.5 (1a 0.2367, 1b 0.2610). Δ, p y dominios son siempre **frente a esa base**, no frente al baseline oficial.
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

## 3. Conclusiones

1. **El filtro temporal tiene señal, pero solo en los dominios históricos y no como filtro duro** (§1). Los gold caen en `T_q` 1.3 veces (1a) y 1.55 veces (1b) más que los no-gold, pero menos de la mitad de los gold tienen un hecho en `T_q`.
2. **TG-RAG como reranker no funciona en TEMPO** (§2). Todas las variantes sin LLM (ablaciones, `T_q` oráculo, emparejamientos, más cobertura, pesos de fusión) quedan por debajo del híbrido o empatan con él. Ni el componente temporal ni el PPR aportan. **Rama cerrada** sin gastar LLM.
3. **Si se retoma**, solo como rasgo "hecho con τ en `T_q`" para un reranker aprendido (cross-encoder), no como reranker propio.
