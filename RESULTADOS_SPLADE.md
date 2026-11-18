# Resultados de SPLADE — estado al 2026-11-18

Resultados de lo implementado de [`PLAN_SPLADE.md`](PLAN_SPLADE.md): recuperación *sparse* aprendida (SPLADE) como retriever propio y como brazo del híbrido. El plan se basó en el repositorio de IIMAS-RAG para SemEval-2026 Task 8 ([github.com/Vania-Janet/mtrag_semeval2026](https://github.com/Vania-Janet/mtrag_semeval2026)). El resto de experimentos (BM25, denso, híbrido, oráculos) está en [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md), y TG-RAG en [`RESULTADOS_TGRAG.md`](RESULTADOS_TGRAG.md).

Reglas de siempre:

- Todo se ajusta en train y dev no se ha tocado.
- Las cifras son **macro nDCG@10 en train** (13 dominios), salvo que se indique otra cosa.
- Una mejora se acepta con Δ ≥ +0.01, p < 0.05 (test pareado de aleatorización) y mejora en ≥ 8 de 13 dominios.
- Las etiquetas de clase e intención de `guidance` solo se usan para analizar (§5.4).

**Estado:**

| Experimento | Estado |
|---|---|
| SP0 · montaje y humo | ✅ §0 |
| SP1 · `splade-ed` solo, 13 dominios | ✅ §1: supera con holgura a BM25 y empata con el denso |
| SP2 · `splade-v3` frente a `splade-ed` | ✅ §2: v3 algo mejor, sin significancia → **se queda `splade-ed`** |
| SP3 · poda de la consulta | ✅ §3: no ayuda → **sin poda** |
| SP4 · fusión | ✅ §4: **nueva base de la primera etapa: BM25 + SPLADE + denso** |
| SP5 · diagnóstico | ✅ §5 |
| SP6 · dev | ⏳ al final, una sola vez, junto con las demás ramas |

**Resumen:**

| | Baseline oficial | Mejor BM25 | Mejor denso | SPLADE solo | Híbrido anterior (BM25 + denso) | **BM25 + SPLADE + denso** |
|---|---:|---:|---:|---:|---:|---:|
| 1a | 0.1075 | 0.1528 | 0.2277 | 0.2232 | 0.2367 | **0.2545** (+0.018, p 0.010, 10/3) |
| 1b | 0.1024 | 0.1660 | 0.2449 | 0.2474 | 0.2610 | **0.2793** (+0.018, p 0.0008, 12/1) |

La configuración nueva es RRF con k = 60 de tres brazos, con estos pesos y runs:

| Brazo | Peso | Run 1a | Run 1b |
|---|---:|---|---|
| BM25 | 0.5 | `rrf(limpia,titulo)` | `titulo+paso_limpio` |
| `splade-ed` | 1 | `limpia` | `rrf(titulo+paso,limpia)` |
| `bge-base` | 1 | `rrf(limpia,titulo)` | `limpia+paso` |

Los runs fusionados se llaman:

- 1a: `fusion/rrf(bm25:rrf(limpia,titulo)+splade-ed:limpia+bge-base:rrf(limpia,titulo))w0.5,1,1`;
- 1b: `fusion/rrf(bm25:titulo+paso_limpio+splade-ed:rrf(titulo+paso,limpia)+bge-base:limpia+paso)w0.5,1,1`.

---

## 0. SP0 · Montaje y recursos (servidor, 2 × RTX 3070)

Las 9 pruebas de [`tests/test_splade.py`](tests/test_splade.py) pasan, incluida la del modelo real:

- `SparseEncoder` da los mismos vectores que la fórmula del repo de IIMAS;
- el vocabulario de `splade-ed` es el de `bge-base`.

Los trozos (254 tokens, solape de 30, como mucho 4 por documento) son **idénticos a los del denso** en los 13 dominios.

| Medida | `splade-ed` | `splade-v3` |
|---|---:|---:|
| Documentos / trozos | 1.167 M / 1.874 M | ídem |
| Codificación con las dos GPUs | 38 min | 42 min |
| Ritmo por GPU (fp16, batch 32) | 390–440 trozos/s | 350–430 trozos/s |
| Términos activos por trozo (media; p99) | 181–209; ~300 | 275–326; ~490–590 |
| Índice | 2.3 GB | 3.5 GB |
| RAM máxima al codificar | 2.6 GB | — |
| Búsqueda de una variante, 13 dominios | ~1–2 min (`history`: 8–30 s) | algo más |

Comprobación de cordura en `iota`, `quant` y `law` (nDCG@10, con R@1000 entre paréntesis). SPLADE supera a BM25 en todas las variantes; con tres dominios no se decide nada.

| Variante | BM25 | `bge-base` | `splade-ed` |
|---|---:|---:|---:|
| 1a `limpia` | 0.078 (0.704) | 0.218 (0.867) | **0.231** (0.936) |
| 1a `rrf(limpia,titulo)` | 0.114 (0.729) | **0.227** (0.858) | 0.194 (0.924) |
| 1b `limpia+paso` | 0.078 (0.753) | **0.231** (0.901) | 0.192 (0.961) |
| 1b `titulo+paso_limpio` | 0.138 (0.886) | 0.190 (0.899) | **0.210** (0.964) |

## 1. SP1 · `splade-ed` solo en los 13 dominios

**Por variante y agregación.** El score de un documento sale de su mejor trozo (`max`), de su primer trozo (`first`) o de un único vector, el máximo de los vectores de sus trozos (`maxpool`). R@1000 es con `max`; Δ, p y dominios son frente al mejor BM25 (1a 0.1528, 1b 0.1660).

| Subtrack | Variante | `max` | `first` | `maxpool` | R@1000 | Δ vs BM25 (`max`) |
|---|---|---:|---:|---:|---:|---|
| 1a | `cruda` | 0.2155 | 0.2156 | 0.1823 | 0.876 | +0.063 (<0.001, 10/3) |
| 1a | **`limpia`** | **0.2232** | 0.2155 | 0.1900 | 0.879 | +0.070 (<0.001, 11/2) |
| 1a | `titulo` | 0.1869 | 0.1889 | 0.1759 | 0.822 | +0.034 (0.002, 10/3) |
| 1a | `titulo+inicio` | 0.2145 | 0.2097 | 0.1856 | 0.856 | +0.062 (<0.001, 10/3) |
| 1a | `rrf(limpia,titulo)` | 0.2209 | 0.2182 | 0.1883 | **0.894** | +0.068 (<0.001, 12/1) |
| 1b | `oficial` | 0.2254 | 0.2241 | 0.1782 | 0.913 | +0.059 (<0.001, 11/2) |
| 1b | `paso` | 0.1514 | 0.1570 | 0.1335 | 0.794 | −0.015 (0.11, 6/7) |
| 1b | `limpia+paso` | 0.2283 | 0.2260 | 0.1823 | 0.913 | +0.062 (<0.001, 11/2) |
| 1b | `titulo+paso` | 0.2330 | 0.2321 | 0.2005 | 0.911 | +0.067 (<0.001, 12/1) |
| 1b | `titulo+paso_limpio` | 0.2322 | 0.2336 | 0.2028 | 0.908 | +0.066 (<0.001, 12/1) |
| 1b | `rrf(paso,limpia)` | 0.2269 | 0.2252 | 0.1828 | 0.931 | +0.061 (<0.001, 11/2) |
| 1b | **`rrf(titulo+paso,limpia)`** | **0.2474** | 0.2458 | 0.2016 | 0.930 | +0.081 (<0.001, 12/1) |

**Frente al denso y al híbrido anterior** (test pareado):

| Comparación | SPLADE | Referencia | Δ | p | +/−dom | R@1000 SPLADE vs ref. |
|---|---:|---:|---:|---:|---|---|
| 1a `limpia` vs `bge-base` `limpia` | 0.2232 | 0.2234 | −0.000 | 0.99 | 6/7 | 0.879 vs 0.872 |
| 1a `limpia` vs mejor denso (`rrf(limpia,titulo)`) | 0.2232 | 0.2277 | −0.005 | 0.73 | 6/7 | 0.879 vs 0.863 |
| 1a `limpia` vs híbrido anterior | 0.2232 | 0.2367 | −0.014 | 0.22 | 4/9 | 0.879 vs 0.870 |
| 1b `rrf(titulo+paso,limpia)` vs `bge-base`, misma variante | 0.2474 | 0.2378 | +0.010 | 0.34 | 9/4 | 0.930 vs 0.926 |
| 1b `rrf(titulo+paso,limpia)` vs mejor denso (`limpia+paso`) | 0.2474 | 0.2449 | +0.003 | 0.84 | 9/4 | 0.930 vs 0.917 |
| 1b `rrf(titulo+paso,limpia)` vs híbrido anterior | 0.2474 | 0.2610 | −0.014 | 0.14 | 4/9 | 0.930 vs 0.938 |

**Complementariedad con el mejor denso** ([`scripts/compare_runs.py`](scripts/compare_runs.py); fracción de los gold de cada consulta, macro):

| | Profundidad | R SPLADE | R denso | R unión | Solo SPLADE | Solo denso | Jaccard top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1a SPLADE `limpia` + denso `rrf(limpia,titulo)` | 100 | 0.618 | 0.641 | **0.739** | **0.098** | 0.121 | 0.21 |
| 1a ídem | 1000 | 0.879 | 0.863 | 0.924 | 0.061 | 0.045 | 0.21 |
| 1a *BM25 `rrf(limpia,titulo)` + denso, para comparar* | 100 | 0.453 | 0.641 | 0.687 | 0.046 | 0.234 | 0.12 |
| 1b SPLADE `rrf(titulo+paso,limpia)` + denso `limpia+paso` | 100 | 0.725 | 0.714 | **0.813** | **0.099** | 0.088 | 0.28 |
| 1b ídem | 1000 | 0.930 | 0.917 | 0.964 | 0.047 | 0.034 | 0.28 |
| 1b *BM25 `titulo+paso_limpio` + denso, para comparar* | 100 | 0.544 | 0.714 | 0.794 | 0.080 | 0.250 | 0.12 |

Lecturas:

- **SPLADE supera con holgura al mejor BM25**: +0.070 en 1a y +0.081 en 1b, p < 0.001, 11–12 de 13 dominios.
- **Solo, empata con el denso.** Tiene algo más de R@1000 y queda por debajo del híbrido anterior (−0.014, no significativo).
- **Quiere la consulta como el denso, no como BM25.**
  - En 1a gana la consulta completa y limpia; el título solo es peor (0.187).
  - En 1b el paso aislado se hunde (0.151).
  - Lo mejor de 1b es fusionar `titulo+paso` con la consulta limpia.
- **Agregación: `max` ≥ `first` ≫ `maxpool`.** `max` y `first` difieren en 0.000–0.008; `maxpool` pierde 0.03–0.05. El vector máximo por documento acumula los términos de todos sus trozos, y así los documentos largos casan con más términos de la consulta (no hay normalización por longitud). **Se descarta `maxpool`.** En el humo de `iota`, con 7 consultas, parecía la mejor: otra razón para no decidir con dominios sueltos.
- **SPLADE complementa al denso mejor que BM25.**
  - En el top-100 aporta un ~10 % de gold que el denso no tiene; BM25 aporta un 4.6 % en 1a y un 8.0 % en 1b.
  - La unión llega a 0.739 (1a) y 0.813 (1b).

## 2. SP2 · `splade-v3` frente a `splade-ed`

Misma variante y agregación `max`:

| Subtrack | Variante | `splade-ed` | `splade-v3` | Δ | p | +/−dom | R@100 ed → v3 | R@1000 ed → v3 |
|---|---|---:|---:|---:|---:|---|---|---|
| 1a | `limpia` | 0.2232 | 0.2294 | +0.006 | 0.28 | 8/5 | 0.618 → 0.621 | 0.879 → 0.888 |
| 1a | `rrf(limpia,titulo)` | 0.2209 | 0.2287 | +0.008 | 0.25 | 8/5 | 0.630 → 0.647 | 0.894 → 0.896 |
| 1b | `rrf(titulo+paso,limpia)` | 0.2474 | 0.2532 | +0.006 | 0.34 | 8/5 | 0.725 → 0.753 | 0.930 → 0.932 |
| 1b | `titulo+paso` | 0.2330 | 0.2463 | +0.013 | 0.051 | 9/4 | 0.693 → 0.713 | 0.911 → 0.914 |

- **`splade-v3` mejora siempre, pero poco**: +0.006 a +0.013 en las 4 comparaciones. Ninguna cumple el criterio completo; la más cercana se queda en p = 0.051.
- Dentro de la fusión de SP4 tampoco aporta (+0.001; §4). **Se queda `splade-ed`**: índice un 35 % más pequeño y descarga libre (`splade-v3` está restringido en Hugging Face).

## 3. SP3 · Poda de la consulta

Distribución de términos activos por consulta (`splade-ed`):

| Variante | n | p10 | p50 | p90 | p99 | > 64 | > 128 | > 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1a `limpia` | 1,211 | 97 | 152 | 236 | 313 | 99 % | 69 % | 6 % |
| 1a `titulo` | 1,211 | 44 | 61 | 82 | 103 | 41 % | 0 % | 0 % |
| 1b `titulo+paso` | 2,762 | 78 | 102 | 129 | 156 | 99 % | 11 % | 0 % |
| 1b `limpia+paso` | 2,762 | 108 | 162 | 244 | 313 | 100 % | 78 % | 7 % |

Cada consulta se poda a sus N términos de más peso; cada N se compara con la misma consulta sin podar. La rejilla del plan se ajustó a {32, 64, 96, 128}: con 256 solo se podaría el 6–7 % de las consultas.

| N | 1a `limpia` (0.2232) | Δ, p, +/−dom | 1b `rrf(titulo+paso,limpia)` (0.2474) | Δ, p, +/−dom |
|---:|---:|---|---:|---|
| 32 | 0.2057 | −0.018, 0.001, 2/11 | 0.2365 | −0.011, 0.014, 3/10 |
| 64 | 0.2179 | −0.005, 0.024, 1/12 | 0.2490 | +0.002, 0.38, 8/5 |
| 96 | 0.2228 | −0.000, 0.75, 5/7 | 0.2455 | −0.002, 0.17, 4/9 |
| 128 | 0.2226 | −0.001, 0.74, 5/5 | 0.2474 | −0.000, 0.99, 4/8 |

- **La poda no ayuda.** Fuerte (32–64 términos) empeora; desde ~96 no cambia nada. Los términos de poco peso no estorban porque SPLADE ya los pondera poco.
- Es lo contrario de BM25, donde acortar la consulta era lo que más rendía.

## 4. SP4 · Fusión

RRF con k = 60. El denso pesa 1 y se varían w_BM25 y w_SPLADE. La referencia es el híbrido anterior (1a 0.2367, 1b 0.2610). La fila w_BM25 = 0 es SPLADE + denso; el resto, tres brazos. En negrita, las celdas que cumplen el criterio de aceptación; entre paréntesis, por qué no lo cumple una celda que mejora.

1a:

| w_BM25 \ w_SPLADE | 0.3 | 0.5 | 0.7 | 0.75 | 1 | 1.5 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.2396 | 0.2402 | 0.2425 | — | 0.2466 | 0.2450 |
| 0.25 | — | 0.2477 (7/6) | — | — | **0.2510** | — |
| 0.5 | — | **0.2504** | — | **0.2517** | **0.2545** | 0.2520 (p 0.06) |
| 0.75 | — | — | — | **0.2517** | **0.2518** | 0.2502 (p 0.10) |
| 1 | — | — | — | 0.2487 (p 0.07) | 0.2485 (p 0.10) | 0.2515 (7/6) |

1b:

| w_BM25 \ w_SPLADE | 0.3 | 0.5 | 0.7 | 0.75 | 1 | 1.5 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.2588 | 0.2647 | 0.2681 | — | 0.2681 | 0.2673 |
| 0.25 | — | **0.2720** | — | — | 0.2715 (p 0.05) | — |
| 0.5 | — | **0.2725** | — | **0.2761** | **0.2793** | **0.2779** |
| 0.75 | — | — | — | **0.2746** | **0.2784** | **0.2776** |
| 1 | — | — | — | 0.2680 | 0.2713 (p 0.07) | **0.2775** |

Además se probó, solo en 1b, añadir un segundo denso (`rrf(titulo+paso,limpia)`) a SPLADE + denso: 0.2597–0.2656, sin mejora sobre el híbrido anterior.

**Configuración elegida: w_BM25 = 0.5, w_SPLADE = 1, w_denso = 1, k = 60.** Es el máximo de los dos subtracks y queda dentro de la meseta (w_BM25 0.5–0.75, w_SPLADE 0.75–1.5), no en un borde. Equivale al híbrido anterior tal cual, más SPLADE como tercer brazo con peso 1.

| Subtrack | Híbrido anterior | **BM25 + SPLADE + denso** | Δ | p | +/−dom | R@100 | R@1000 |
|---|---:|---:|---:|---:|---|---|---|
| 1a | 0.2367 | **0.2545** | +0.018 | 0.010 | 10/3 | 0.657 → 0.690 | 0.870 → **0.904** |
| 1b | 0.2610 | **0.2793** | +0.018 | 0.0008 | 12/1 | 0.759 → 0.795 | 0.938 → **0.951** |

Comprobaciones con la configuración elegida (Δ frente a ella misma con `splade-ed` y k = 60):

| Cambio | 1a | 1b |
|---|---|---|
| `splade-v3` en lugar de `splade-ed` | 0.2552 (+0.001, p 0.85) | 0.2801 (+0.001, p 0.80) |
| k = 20 | 0.2572 (+0.003, p 0.36) | 0.2793 (+0.000, p 0.99) |
| k = 100 | 0.2519 (−0.003, p 0.19) | 0.2765 (−0.003, p 0.06, 3/10) |

Lecturas:

- **SPLADE entra en la primera etapa.** El híbrido de tres brazos supera al anterior en los dos subtracks, con significancia y en 10–12 de 13 dominios. Frente al baseline oficial es 2.4 veces en 1a y 2.7 veces en 1b.
- **SPLADE no sustituye a BM25: se suma.** SPLADE + denso mejora poco y sin significancia (+0.010 en 1a, +0.007 en 1b); con los tres brazos se gana el doble. BM25 aporta el emparejamiento exacto y SPLADE la ponderación aprendida y la expansión.
- **El recall también sube** (R@100 +0.03–0.04; R@1000 de 0.870 a 0.904 en 1a), y eso amplía el techo del futuro reranker.
- **k = 60 es estable**, y `splade-v3` no aporta dentro de la fusión.
- **Cautela.** Se probaron ~17 combinaciones de pesos por subtrack sin corregir por comparaciones múltiples.
  - A favor: toda la región de tres brazos con w_BM25 ≤ 0.75 supera al híbrido anterior (1a +0.011 a +0.018; 1b +0.010 a +0.018), y el punto elegido es interior.
  - Aun así, el máximo de train es algo optimista (sus vecinos dan ~+0.015). La confirmación será dev (SP6).

## 5. SP5 · Diagnóstico (sin decisiones)

[`scripts/splade_diag.py`](scripts/splade_diag.py) (`expansiones`, `anios`, `grupos`, `ejemplos`). Informes completos en `outputs/exp/splade/diag/*.md` del servidor.

### 5.1 Expansiones de la consulta

Media por consulta (`splade-ed`):

| | Términos activos | No están en el texto (% de términos) | Su peso (% del total) | Expansiones en el top-20 | Añade un término temporal en el top-20 |
|---|---:|---:|---:|---:|---:|
| 1a `limpia` | 160 | 65 % | 42 % | 17 % | 16 % |
| 1b `titulo+paso` | 103 | 81 % | 57 % | 37 % | 36 % |

- **Los términos fuertes son las palabras de la propia consulta.** La expansión es una cola de términos de menos peso, y pesa más en las consultas cortas (1b).
- **Las expansiones son temáticas y morfológicas**: entidades (`wwii`, `germany`, `russia`, `rome`, `coin`) y variantes (`badges`, `pins`, `mutation`, `generation`). Hay algo de ruido (`because`, `yes`, `long`).
- **Los términos temporales que añade son genéricos**: `early`, `date`, `era`, `earlier`, `origin`, `period`. Casi nunca añade periodos concretos (`1800s` 3 veces, `19th` 2 en 1a). Los añade más en MAP (29 % en 1a, 43 % en 1b) y en TPC (22 % y 47 %).

### 5.2 Años

- **Vocabulario de BERT:** solo 398 de los 1,030 años entre 1000 y 2029 son un único token. Son casi todos los de 1700–2021. Antes de ~1540 casi ninguno lo es, **y 2022–2029 tampoco**: se parten en *wordpieces*. Por eso SPLADE no puede emparejar como término exacto los años medievales ni los más recientes, justo los de los dominios de criptomonedas y de política actual.
- Tienen algún año de un solo token el 38 % de las consultas de 1a y el 16 % de los pasos de 1b. En esas consultas:

| | Año propio activo | Año propio en el top-20 (rango mediano) | Añade otros años | Distancia mediana de los años añadidos | Añade la década (1834 → `1830s`) | Añade el siglo (`19th`) |
|---|---:|---:|---:|---:|---:|---:|
| 1a `limpia` | 90 % | 35 % (30) | 84 % (mediana: 3 años) | 2 años | 14 % | 1 % |
| 1b `titulo+paso` | 98 % | 80 % (8) | 88 % (mediana: 3 años) | 2 años | 15 % | 0 % |

- **SPLADE amplía los años a sus vecinos** (±2 años), una forma de tolerancia temporal. Pero **casi nunca pasa a una granularidad más gruesa** (década o siglo). La normalización de anclas de K4 ([`PLAN_EXPERIMENTOS.md`](PLAN_EXPERIMENTOS.md) §6.2) no está hecha de forma implícita.
- **¿Activan los gold el año de la consulta más que los no-gold del top-100 de SPLADE?** En 1a, 0.213 frente a 0.194 (lift 1.10); en 1b, 0.462 frente a 0.299 (lift 1.55).
  - Es la misma señal débil de TG0 ([`RESULTADOS_TGRAG.md`](RESULTADOS_TGRAG.md) §1: 1.3 y 1.55).
  - Es más fuerte en `history`, `hsm`, `economics` y `genealogy` (1b).
  - Es al revés en `bitcoin` y `monero` (lift 0.2–0.6): las páginas no-gold también traen años.

### 5.3 Sin expansión de la consulta (`-noexp`)

Solo se quitan las expansiones de la consulta; los documentos conservan las suyas.

| | Con expansión | Sin expansión | Δ (p, +/−dom) |
|---|---:|---:|---|
| 1a SPLADE solo (`limpia`) | 0.2232 | 0.1968 | −0.026 (<0.001, 1/12) |
| 1b SPLADE solo (`rrf(titulo+paso,limpia)`) | 0.2474 | 0.2217 | −0.026 (<0.001, 2/11) |
| 1a fusión de tres brazos | 0.2545 | 0.2465 | −0.008 (0.014, 4/9) |
| 1b fusión de tres brazos | 0.2793 | 0.2739 | −0.005 (0.08, 5/8) |

- **La ganancia de SPLADE sobre BM25 tiene dos partes.**
  - La ponderación aprendida, junto con la expansión de los documentos: sin expansión de la consulta, SPLADE ya da 0.197 en 1a y 0.222 en 1b, frente a 0.153 y 0.166 de BM25.
  - La expansión de la consulta: otros +0.026.
- **En la fusión, la expansión de la consulta explica ~45 % (1a) y ~30 % (1b) de la mejora sobre el híbrido anterior.**

### 5.4 Por clase de razonamiento e intención

nDCG@10 medio por consulta dentro de cada grupo (no es la macro por dominio). Las etiquetas salen de `guidance`.

Clases:

- EAL: `event_analysis_and_localization`
- EVA: `event_verification_and_authenticity`
- MAP: `materials_artifacts_and_provenance`
- OEC: `origins_evolution_comparative_analysis`
- TPC: `time_period_contextualization`
- TCP: `trends_changes_and_cross_period`

| Grupo | n 1a / 1b | Denso 1a / 1b | SPLADE 1a / 1b | Híbrido anterior 1a / 1b | **Híbrido + SPLADE 1a / 1b** |
|---|---|---|---|---|---|
| EAL | 435 / 974 | 0.237 / 0.244 | 0.259 / 0.279 | 0.260 / 0.278 | **0.281 / 0.299** |
| EVA | 80 / 179 | 0.277 / 0.303 | 0.291 / 0.308 | 0.264 / 0.309 | **0.286 / 0.321** |
| MAP | 68 / 145 | 0.288 / 0.289 | 0.249 / 0.294 | 0.277 / 0.305 | **0.305 / 0.328** |
| OEC | 171 / 455 | 0.213 / 0.237 | 0.208 / 0.252 | 0.238 / 0.254 | **0.252 / 0.277** |
| TPC | 256 / 604 | 0.231 / 0.232 | 0.251 / 0.261 | 0.238 / 0.257 | **0.266 / 0.284** |
| TCP | 117 / 264 | 0.174 / 0.198 | 0.151 / 0.205 | 0.195 / 0.225 | **0.196 / 0.245** |
| `duration` | 101 / 202 | 0.176 / 0.167 | 0.139 / 0.172 | 0.159 / 0.173 | **0.175 / 0.183** |
| `ongoing_status` | 134 / 283 | 0.228 / 0.247 | 0.260 / 0.251 | 0.228 / 0.240 | **0.255 / 0.275** |
| `order` | 84 / 188 | 0.202 / 0.220 | 0.233 / 0.275 | 0.226 / 0.296 | **0.268 / 0.321** |
| `period_definition` | 244 / 597 | 0.242 / 0.258 | 0.258 / 0.278 | 0.255 / 0.283 | **0.287 / 0.309** |
| `when` | 277 / 624 | 0.265 / 0.269 | 0.264 / 0.293 | 0.298 / 0.301 | **0.298 / 0.319** |

- **El híbrido con SPLADE es el mejor en todos los grupos de 1b y en casi todos los de 1a.** Las excepciones de 1a son de menos de 0.01: SPLADE solo en EVA y en `ongoing_status`; el denso en `duration` y en la clase `other`.
- **Mayores ganancias sobre el híbrido anterior:** `order` (+0.04 en 1a), `period_definition` (+0.03), `ongoing_status`, TPC y MAP (+0.03).
- **SPLADE solo supera al denso en los grupos con anclas concretas**: EAL, TPC, EVA, `order`, `ongoing_status` y `period_definition`.
- **Siguen siendo difíciles** TCP (0.196 en 1a) y `duration` (0.175 en 1a), donde SPLADE apenas ayuda en 1a.

### 5.5 Ejemplos (1a)

- Hay 52 pares (consulta, gold) que SPLADE pone en su top-10 y el denso fuera de su top-100. Al revés hay 74.
- **SPLADE gana con términos concretos o poco frecuentes**: *time warp attack* → algoritmo de ajuste de dificultad; *penal colonies* → Juan exiliado en Patmos; *monarcas emparentados* → foto de los nueve soberanos en el funeral de Eduardo VII.
- **El denso gana con paráfrasis**: *view only wallet* ↔ *watch-only wallet*; *boil water before metal pots* ↔ *stone boiling*.

---

## 6. Conclusiones

1. **Nueva base de la primera etapa:** RRF de BM25 (w = 0.5), `splade-ed` (w = 1) y `bge-base` (w = 1), con k = 60. Da 0.2545 en 1a y 0.2793 en 1b: +0.018 sobre el híbrido anterior, significativo en los dos subtracks. Es 2.4–2.7 veces el baseline oficial y sube el recall (R@1000 0.904 en 1a y 0.951 en 1b).
2. **SPLADE solo empata con el denso**, pero lo complementa mejor que BM25. Encuentra gold distintos: términos concretos frente a paráfrasis.
3. **SPLADE se comporta como el denso respecto a la consulta.** Quiere el texto completo y limpio, y podarlo no ayuda. Es lo contrario de BM25.
4. **Se queda `splade-ed`.** `splade-v3` es algo mejor solo, pero no significativamente, y no aporta en la fusión. Se descarta `maxpool`.
5. **La aportación temporal de SPLADE es indirecta.**
   - Amplía años a sus vecinos y añade palabras temporales genéricas.
   - No normaliza a décadas ni siglos.
   - No puede emparejar como un solo token los años anteriores a ~1540 ni los de 2022 en adelante.
   - La ganancia viene sobre todo de la ponderación aprendida y de la expansión temática.

## 7. Pendiente / siguientes pasos

1. **SP6:** validar la nueva base en dev, una sola vez y al final, junto con las demás ramas.
2. **Repetir O4** (enrutamiento por clase) con candidatos de verdad distintos: BM25, denso, SPLADE y el híbrido.
3. **K4:** normalizar los años a décadas y siglos. SPLADE no lo hace, así que sigue abierto. Encaja con los años que no son un token (antes de ~1540 y desde 2022).
4. **Fase 5 (reescritura con LLM):** probar cada reescritura también sobre el brazo SPLADE. En IIMAS fue lo que más aportó.
5. **Grupos más flojos:** TCP y `duration`. Son candidatos a la descomposición por periodo (Q6).
