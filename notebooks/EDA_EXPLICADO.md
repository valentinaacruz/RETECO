# EDA del Track 1 (TEMPO): qué se hizo y cómo leer los resultados

Esta guía acompaña al notebook [`notebooks/eda_track1_tempo.ipynb`](notebooks/eda_track1_tempo.ipynb). Para cada sección explica:

1. **qué se hizo y por qué**;
2. **cómo leer** cada tabla o gráfico;
3. **qué significan** los resultados obtenidos y qué decisión sugieren.

Al final hay un **glosario** con todos los conceptos técnicos (nDCG, recall, BM25, AUC, bootstrap…). Si un término no te suena, búscalo allí.

> **Nota (reorganización de notebooks):** el notebook original se dividió en dos. `eda_track1_tempo.ipynb` conserva el análisis de datos y se renumeró (consultas = sección 3, pasos = 4, guidance = 5, hallazgos = 6). Todo lo que ejecuta un retriever (secciones 3 y 7 de esta guía: BM25 oficial, formulaciones de 1b, rendimiento por clase temporal, densos, varianza, revisión de fallos) pasó a `baseline_track1_tempo.ipynb`, que además compara con el paper. Los números de sección de esta guía siguen la numeración antigua.

> Todas las cifras son de **train**. Dev solo se usó en la sección 1, para comprobar que los archivos estén bien formados, nunca para medir rendimiento. Así dev sigue siendo una validación "limpia" para el final.

---

## Índice

0. [La tarea en dos minutos](#0-la-tarea-en-dos-minutos)
1. [Inventario y estructura](#1-inventario-y-estructura-sección-1-del-notebook)
2. [Corpus](#2-corpus-sección-2)
3. [Infraestructura: el BM25 oficial](#3-infraestructura-el-bm25-oficial-sección-3)
4. [Consultas 1a](#4-consultas-1a-sección-4)
5. [Pasos 1b](#5-pasos-1b-sección-5)
6. [Guidance temporal](#6-guidance-temporal-sección-6)
7. [Baselines y diagnóstico de errores](#7-baselines-y-diagnóstico-de-errores-sección-7)
8. [Resumen: qué nos dice todo esto](#8-resumen-qué-nos-dice-todo-esto)
9. [Glosario](#9-glosario)
10. [Detalles prácticos del notebook](#10-detalles-prácticos-del-notebook)

---

## 0. La tarea en dos minutos

Tenemos 13 **dominios** (bitcoin, history, law…). Cada dominio tiene su propio **corpus** de documentos y un conjunto de **consultas** sacadas de StackExchange. Para cada consulta, los organizadores marcaron qué documentos son relevantes: son los **gold**, y están en los archivos **qrels**.

- **1a:** dada la consulta completa, devolver un ranking de documentos con los gold lo más arriba posible.
- **1b:** la consulta viene descompuesta en **pasos** (por ejemplo, "Consult Bitcoin Core documentation…"), y hay que hacer un ranking para cada paso.

La métrica oficial es **nDCG@10** (ver glosario): mide lo bien colocados que están los gold entre los 10 primeros resultados. Se calcula por consulta, se promedia dentro de cada dominio y después se promedian los 13 dominios (**macro-media**). Consecuencia importante: `iota`, con 7 consultas de train, pesa en la nota final lo mismo que `history`, que tiene 561.

---

## 1. Inventario y estructura (sección 1 del notebook)

### Qué se hizo
Antes de analizar nada, hay que comprobar que los datos son lo que dicen ser. Si hubiera IDs rotos o qrels inconsistentes, cualquier conclusión posterior sería falsa. Se revisó:

| Comprobación | Pregunta que responde |
|---|---|
| **1.1 Conteos** | ¿Coinciden los archivos con lo que declara `split_manifest.json`? |
| **1.2 Esquemas** | ¿Tiene cada registro los campos esperados, con el tipo correcto y sin nulos? |
| **1.4 Integridad de IDs** | ¿Existe cada gold en el corpus? ¿Coinciden los qrels con los `gold_ids`? ¿Tienen los `step_id` el formato correcto? ¿Aparece algún ID eliminado por la deduplicación? |
| **1.5 Split** | ¿Hay alguna consulta a la vez en train y en dev? |
| **1.6 Relación 1a ↔ 1b** | ¿Tienen todas las consultas pasos? ¿Por qué faltan números de paso? |

### Cómo leerlo
- **Tabla de conteos (1.1):** una fila por dominio. La columna `ok_manifest` vale `True` si todos los conteos de ese dominio coinciden con el manifiesto.
- **Gráfico de barras (1.1):** cada barra es la fracción de consultas de train de un dominio. La línea discontinua marca 1/13 ≈ 0.077, que es lo que pesa cada dominio en la macro-media. Una barra a la izquierda de la línea indica un dominio con pocas consultas que pesa más de lo que su tamaño sugiere. Una barra a la derecha (solo `history`) indica lo contrario.
- **Tabla de esquemas (1.2):** solo lista los problemas encontrados. `vacío` significa que el campo existe pero es una cadena o lista vacía.
- **Tabla de anomalías (1.7):** si dice "No se encontraron anomalías estructurales", todos los chequeos de integridad pasaron.

### Qué significan los resultados
- **Los datos están limpios.** Todo coincide con el manifiesto (1,167,159 documentos, 1,730 consultas, 3,976 pasos), no hay solapamiento train/dev y hay **cero anomalías** de integridad.
- Los únicos campos vacíos están en **guidance**. Por ejemplo, 36 consultas de train no tienen clase temporal primaria y 23 no tienen `retrieval_plan`. No es un error, pero hay que tenerlo en cuenta si se usa guidance para entrenar.
- **No todas las consultas tienen pasos:** 90 de 1,211 (7.4 %) no aparecen en `steps_train.jsonl`. En `workplace` solo el 64 % tiene pasos.
- **Hay "pasos hueco"**: por ejemplo, una consulta solo tiene `step3`. La sección 5.5 explica por qué.

> **Interpretación práctica:** podemos confiar en los archivos. Cuando el sistema rinda mal, no será por datos corruptos.

---

## 2. Corpus (sección 2)

### 2.1 Longitud de los documentos

**Qué se hizo.** Se midió cada documento en caracteres, en palabras y, con una muestra, en **tokens** del tokenizador del modelo BGE. Los modelos neuronales tipo BERT solo leen los primeros **512 tokens**; el resto se ignora (se trunca).

**Cómo leerlo.**
- *Tabla de percentiles:* `p50` es la mediana (la mitad de los documentos son más cortos). `p90` significa que el 90 % de los documentos son más cortos que ese valor. `p99` y `max` muestran la cola de documentos larguísimos.
- *Diagrama de cajas (escala logarítmica):* la línea dentro de la caja es la mediana, la caja va del percentil 25 al 75 y los "bigotes" muestran el rango típico. Las marcas del eje (1, 10, 100, 1k, 10k) crecen ×10 cada vez.
- *Tabla de tokens:* `tokens/palabra` indica cuántos tokens produce una palabra en promedio. `%>512` es el porcentaje de documentos que el modelo **no puede leer enteros**.

**Qué significa.**
- Un documento típico tiene ~150 palabras (≈ 200 tokens), lo cual es manejable.
- Pero entre el **9 % y el 18 %** de los documentos superan los 512 tokens, según el dominio. En `history` el p90 llega a 1,095 tokens.
- Cada palabra produce ~1.45 tokens.

> **Decisión:** usar **chunking**, es decir, partir los documentos largos en trozos de ~256–384 tokens con algo de solapamiento. Cada trozo se puntúa por separado y el documento recibe la nota de su mejor trozo. Si no, la información que esté al final de un documento largo es invisible para el modelo.

### 2.2 Documentos vacíos, boilerplate y casi-duplicados

**Qué se hizo.**
- **Casi vacíos:** documentos con menos de 10 palabras.
- **Boilerplate:** texto repetido que no aporta contenido (botones de "compartir en Facebook", avisos de Wikipedia…). Se detectó buscando los primeros 80 caracteres que se repiten en muchos documentos.
- **Casi-duplicados:** la v1.1 eliminó las copias *idénticas*, pero no las que difieren en una palabra o una fecha. Para encontrarlas se usó **MinHash + LSH** (ver glosario), que estima rápido la similitud de Jaccard entre todos los pares sin compararlos uno a uno. Se aplicó en 4 dominios pequeños.

**Cómo leerlo.**
- `casi_vacios` y `%`: cuántos documentos son casi vacíos. `gold_casi_vacios` indica cuántos de ellos son gold.
- Tabla de prefijos: `docs` es cuántos documentos empiezan igual, `dominios` en cuántos dominios aparece ese inicio y `gold_train` cuántos de esos documentos son gold.
- Tabla de MinHash: `pares_J≥0.8` es el número de pares con similitud ≥ 0.8; `%docs` es el porcentaje del corpus que tiene al menos un "gemelo"; `pares_gold–no_gold` cuenta los pares en los que uno es gold y el otro no.

**Qué significa.**
- Los casi vacíos (0.06–0.32 %) y el boilerplate (≈ 0.2 % del corpus) son **pocos** y casi nunca gold. No son un problema grave.
- Los **casi-duplicados sí son frecuentes:** entre el 9 % y el 11 % de los documentos de los dominios pequeños tiene un gemelo con Jaccard ≥ 0.8. El ejemplo mostrado son dos posts del mismo blog con la misma plantilla.
- Los pares gold–no-gold (34 en `law`) son un caso curioso: un texto casi igual a un gold **no** está marcado como relevante. Si el sistema lo pone arriba, la métrica lo penaliza aunque el contenido sea prácticamente el mismo.

> **Decisión:** probar a **diversificar el top-10**, evitando poner dos casi-gemelos, para no gastar posiciones en contenido repetido.

### 2.3 Formato, idioma y caracteres raros

**Qué significa.** El corpus es **texto plano** en inglés: HTML ≈ 0 % y algo de markdown, hasta un 11 % en `bitcoin`. Los documentos con muchos caracteres no ASCII son casos raros: listas de idiomas de los menús de Wikipedia, spam en chino o ruso en `bitcoin` y algún texto en griego. Son menos del 0.6 % en cualquier dominio.

> **Decisión:** no hace falta un pipeline de limpieza complejo. Basta con normalizar el markdown y los restos de fórmulas (ver 2.6).

### 2.4 Estructura de los IDs y posible **leakage** ⚠️

Este es uno de los hallazgos más importantes.

**Qué se hizo.** Los IDs tienen dos formatos:

| Formato | Ejemplo | Cuántos |
|---|---|---|
| **corto** | `law/051ef0b9_0001.txt` | 386,031 |
| **largo** | `law/104884_eae0d585d917be2d_134.txt` | 781,128 (67 % del corpus) |

Se comprobó: (a) qué formato tienen los gold, (b) si el número inicial de los IDs largos (`104884`) coincide con el número de alguna consulta (`104884_xxx`) y (c) si los gold de una consulta tienen números finales consecutivos (`_0001`, `_0002`, `_0003`…).

**Cómo leerlo.**
- Tabla "Formato de ID vs. gold": cruza el formato con si el documento es gold o no.
- `%prefijo=consulta_train` y `%prefijo=consulta_dev`: porcentaje de documentos largos cuyo prefijo es el número de una consulta de train o de dev. Si las dos columnas suman ~100 %, **todos** los documentos largos están asociados a alguna consulta.
- `% consultas con gold consecutivos`: el porcentaje de consultas cuyos gold tienen números finales seguidos.

**Qué significa.**
- **Los 6,098 gold tienen ID corto, y ningún documento largo es gold.** Los documentos largos llevan el número de una consulta (de train o de dev). Parecen páginas que se recuperaron para cada pregunta al construir el dataset y que no se marcaron como relevantes; serían "negativos" de esa consulta. `history` no tiene ningún documento largo.
- Entre el 74 % y el 100 % de las consultas tienen gold con números finales **consecutivos**.
- Todo esto es un **artefacto de cómo se construyó el dataset**, no algo del contenido. Un sistema que lo aprovechara (por ejemplo, descartando todos los IDs largos) quitaría dos tercios del corpus sin perder ningún gold. En la sección 2.6 se ve que, solo con este rasgo, el AUC para detectar gold pasa de 0.64 a 0.90.

> **Decisión:** **no explotarlo.** El test puede construirse de otra manera, y además sería hacer trampa respecto al espíritu de la tarea. Sí es legítimo usar en *entrenamiento* los documentos largos de una consulta de train como **hard negatives**: documentos del mismo tema que no son la respuesta, muy útiles para entrenar un reranker. Conviene preguntar a los organizadores si el test tendrá la misma estructura.

### 2.5 Densidad temporal: gold vs. no-gold

**Qué se hizo.** Con expresiones regulares se contaron años (1500–2039), fechas, meses con número ("March 1834"), décadas y siglos ("1970s", "19th century"), y expresiones relativas ("before", "since", "recently"…). Se compararon los documentos gold de train con el resto.

**Cómo leerlo.** En el gráfico de puntos, cada fila es un dominio: el punto **azul** son los gold y el **naranja** los no-gold. Si los puntos están separados, el rasgo distingue gold de no-gold en ese dominio. Si los puntos se cruzan de lado según el dominio, el rasgo no es fiable como señal general.

- `pct_con_año`: porcentaje de documentos con al menos un año.
- `densidad_temporal`: expresiones temporales por cada 100 palabras.

**Qué significa.**
- **No hay un patrón universal.** En `hsm` (85 % vs. 40 % con año), `politics` y `genealogy`, los gold mencionan años mucho más que el resto. En `travel` (19 % vs. 38 %), `quant` y `monero` ocurre lo contrario.
- La longitud tampoco es consistente: en `hsm` los gold son mucho más largos (747 vs. 170 palabras de mediana), en `bitcoin` y `travel`, más cortos.

> **Interpretación:** que un documento "hable de fechas" no lo convierte en relevante. La temporalidad de la tarea no se resuelve con un filtro de "documentos con años", sino entendiendo **qué periodo** pide la consulta y **qué periodo** cubre el documento.

### 2.6 ¿Son distinguibles los gold por rasgos superficiales?

**Qué se hizo.** Se entrenó una regresión logística para adivinar si un documento es gold usando solo rasgos superficiales (longitud, densidad temporal, formato…), y se midió con **AUC** (ver glosario). Además, se calcularon los términos más característicos de los gold frente a los no-gold con **log-odds** (ver glosario).

**Cómo leerlo.**
- **AUC = 0.5** significa que el modelo acierta como una moneda al aire; **AUC = 1.0**, que separa perfectamente.
- En la tabla de términos, `z_gold` alto indica una palabra mucho más frecuente en los gold. `z_no_gold` muy negativo indica una palabra mucho más frecuente en los no-gold. Un |z| mayor que ~2 ya es significativo, así que valores de 20–38 son diferencias enormes (aunque la muestra también es grande).

**Qué significa.**

| Rasgos | AUC | Lectura |
|---|---:|---|
| Superficiales (longitud, temporalidad, formato) | 0.64 | señal débil |
| Solo longitud | 0.60 | casi nada |
| Solo densidad temporal | 0.57 | casi nada |
| Superficiales + formato de ID | **0.90** | el ID "delata" a los gold (ver 2.4) |

- **Términos de los gold:** `displaystyle`, `frac`, `span`, `mi`, `lt` son **restos de fórmulas matemáticas** (MathML/LaTeX de Wikipedia). Los demás (`election`, `trump`, `soviet`, `war`, `roman`) reflejan los temas de las consultas.
- **Términos de los no-gold:** `url`, `www`, `https`, `journal`, `papers`, `university`… Son bibliografías, listados de publicaciones y páginas institucionales.

> **Interpretación:** con rasgos superficiales no se identifican los gold; hace falta **entender el contenido**. Y conviene **limpiar los restos de fórmulas**, porque son ruido que puede confundir tanto a BM25 como a los modelos neuronales.

---

## 3. Infraestructura: el BM25 oficial (sección 3)

### Qué se hizo
Para medir el rendimiento (secciones 4–7) hacía falta un buscador de referencia. Se usó **BM25** (ver glosario), el mismo que el baseline oficial. Al revisar el código oficial (`starter_kit/official_baseline.py`) aparecieron tres detalles que no son obvios:

1. No usa el índice estándar de Lucene, sino **gensim `LuceneBM25Model`**, que aplica la fórmula BM25 **también a la consulta**: una palabra repetida muchas veces en la consulta pesa más, pero de forma saturada.
2. En 1a usa la consulta **cruda, con HTML** (`<p>`, `<a href=…>`…).
3. En 1b usa `consulta + "\n\nStep: " + instrucción_del_paso`.

La clase `OfficialStyleBM25` reimplementa esto con matrices dispersas (scipy), que consumen menos memoria para los 200k documentos de `history`.

### Por qué importa
Al principio probé el BM25 estándar de Pyserini y daba 0.078 en 1a, en vez del 0.1075 oficial. Con la reimplementación fiel, **las cifras coinciden exactamente** (sección 7.1). Eso garantiza que comparamos contra el mismo punto de partida que el resto de participantes.

### Variantes de consulta
Además de la oficial, se probaron variantes para entender qué ayuda:

| Variante | Texto de búsqueda |
|---|---|
| `1a_oficial` | consulta cruda con HTML |
| `1a_limpia` | título + cuerpo sin HTML |
| `1a_titulo` | solo el título |
| `1b_oficial` | consulta cruda + "Step:" + instrucción |
| `1b_paso` | solo la instrucción |
| `1b_consulta` | solo la consulta (igual para todos sus pasos) |
| `1b_limpia+paso` | consulta sin HTML + instrucción |

---

## 4. Consultas 1a (sección 4)

### 4.1 Limpieza del HTML y longitud

**Qué significa.**
- Las consultas son **largas**: mediana de 127 palabras ya sin HTML, p90 de 321 y p99 de 673. Para un buscador es mucho: un usuario típico escribe 3–10 palabras.
- El HTML ocupa entre el 4 % y el 21 % de los caracteres. En `politics` y `history`, más del 68 % de las consultas tienen enlaces.

> **Decisión:** siempre **quitar el HTML**. Con el BM25 estándar, limpiar subió el nDCG@10 de 0.078 a 0.098. También conviene probar versiones más cortas (título + primera parte del cuerpo) o reformuladas, porque los términos importantes se diluyen entre tanto texto (lo confirma la revisión manual, 7.5).

### 4.2 Número de gold por consulta

**Cómo leerlo.** El histograma muestra cuántas consultas tienen 1, 2, 3… gold; la barra "20+" agrupa las que tienen 20 o más.

**Qué significa.** La mayoría tiene entre 2 y 4 gold. Solo `history` tiene casos extremos (hasta 39), y el 7.5 % de sus consultas tiene más de 10. En esos casos es imposible meter todos los gold en el top-10, pero el nDCG@10 sigue pudiendo llegar a 1, porque se compara con el mejor top-10 posible.

### 4.3 `gold_answers`

**Qué significa.** Cada consulta tiene exactamente **una** respuesta aceptada de StackExchange, de 92 a 436 palabras de mediana según el dominio. **No estará en test**, así que no se puede usar al buscar. Sí sirve para entrenar o como inspiración para HyDE, una técnica en la que un LLM escribe una respuesta hipotética y se busca con ella (ver 4.5).

### 4.4 Expresiones temporales en la consulta

**Cómo leerlo.** Es el porcentaje de consultas de cada dominio con cada tipo de expresión. "alguna" significa que tiene al menos una.

**Qué significa.** El ~80 % de las consultas tiene alguna marca temporal explícita. Las más comunes son las **relativas** ("before", "since", "after"), en más del 60 % de las consultas. Los **años** concretos dependen mucho del dominio: el 90 % en `genealogy`, el 62 % en `politics` y casi ninguno en `cardano` o `iota`. Las consultas con "recencia" ("currently", "still", "latest") son importantes: piden el estado **actual** de algo, y ahí un documento antiguo puede ser incorrecto aunque hable del tema.

### 4.5 Solapamiento léxico consulta ↔ gold

**Qué se hizo.** Se midió la **similitud de Jaccard** (ver glosario) entre las palabras de la consulta y las de cada gold, y se comparó con documentos al azar del mismo dominio y con la respuesta aceptada. Además, se miró en qué posición del ranking BM25 aparece el **primer** gold.

**Cómo leerlo.**
- *Histogramas:* la curva naranja (consulta↔no-gold) está pegada a 0, y la azul (consulta↔gold) está algo más a la derecha. Cuanto más separadas estén, más fácil es para un buscador léxico distinguir los gold.
- *Tabla de posiciones:* para cada dominio, el porcentaje de consultas cuyo primer gold aparece en la posición 1, entre la 2 y la 10, etc. La columna `>1000` indica las consultas en las que BM25 **no encuentra ningún gold** entre los 1000 primeros.

**Qué significa.**
- Los gold comparten **más** vocabulario con la consulta que un documento cualquiera (Jaccard ~0.07 frente a ~0.025), pero en valor absoluto es **muy poco**: comparten aproximadamente 7 de cada 100 términos distintos.
- La respuesta aceptada se parece a los gold solo **un poco más** que la propia consulta. HyDE podría ayudar, pero no es una solución mágica.
- **En `history`, el 44.6 % de las consultas no tiene ningún gold en el top-1000 de BM25.** En `bitcoin`, `quant` y `travel` pasa en ~30 % de los casos. Para esas consultas, ningún reranker aplicado después de BM25 puede arreglar nada, porque solo reordena lo que BM25 ya encontró.

> **Interpretación:** es un problema de **desajuste de vocabulario**: el gold habla del tema con otras palabras. Esto pide recuperación **semántica** (modelos densos) y no solo léxica.

---

## 5. Pasos 1b (sección 5)

### 5.1 y 5.2 Forma de los pasos

**Qué significa.**
- Hay entre 1.8 y 2.6 pasos por consulta, y cada paso tiene entre 1.2 y 1.7 gold de media.
- Las instrucciones son **imperativas**, nunca preguntas (0 % terminan en "?"), y muy formulaicas: 15 verbos ("identify", "search", "analyze", "cross-reference"…) inician el 97 % de los pasos.
- Son cortas: de 13 a 17 palabras.

> **Interpretación:** palabras como "identify", "consult", "documentation" o "historical accounts" no aparecen en los documentos relevantes y meten ruido en la búsqueda. Puede valer la pena **reformular** el paso como pregunta o quitarle esos verbos.

### 5.3 ¿Basta la instrucción sola?

**Cómo leerlo.** Tabla de nDCG@10 por dominio para cada variante; la columna `mejor` indica la ganadora en cada fila.

**Qué significa.**

| Variante | Macro nDCG@10 | Macro R@100 |
|---|---:|---:|
| paso solo | 0.110 | 0.407 |
| consulta sola | 0.102 | 0.386 |
| oficial (consulta + paso) | 0.102 | 0.392 |
| consulta limpia + paso | **0.111** | **0.416** |

- **El paso solo es la mejor variante en 8 de 13 dominios.** Concatenarlo con la consulta larga (como hace el oficial) **diluye** la información específica del paso.
- En `cardano`, `hsm`, `politics`, `bitcoin` y `travel` gana añadir la consulta, que aporta contexto.

> **Decisión:** en lugar de concatenar textos, buscar por separado con el paso y con la consulta y **fusionar los rankings** (por ejemplo con RRF, ver glosario) o ponderarlos. Así se aprovechan ambas señales sin que una ahogue a la otra.

### 5.4 Solapamiento de gold

**Cómo leerlo.**
- `jaccard_entre_pasos`: cuánto se parecen los gold de dos pasos de la misma consulta (0 = ninguno en común, 1 = idénticos).
- `%gold_1b_en_1a`: qué porcentaje de los gold de los pasos están también en los gold de la consulta completa.
- `%gold_1a_cubierto_por_1b`: qué porcentaje de los gold de la consulta aparece en algún paso.

**Qué significa.**
- **Los gold de 1b son siempre un subconjunto de los de 1a** (100 %). No hay documentos "nuevos" en los pasos.
- Los pasos de una misma consulta comparten parte de sus gold (Jaccard 0.2–0.5).
- Entre el 67 % y el 91 % de los gold de 1a aparecen en algún paso.

> **Interpretación:** 1a y 1b están muy conectadas. Un buen sistema de 1a ya da buenos candidatos para 1b, y los rankings de los pasos pueden ayudar a 1a (por ejemplo, fusionándolos).

### 5.5 Pasos "hueco"

**Qué se hizo.** Se comprobó la hipótesis de que los pasos vienen del `retrieval_plan` de guidance y que solo se conservan las acciones que tienen gold.

**Qué significa.** **Hipótesis confirmada al 100 %.** Cada paso coincide exactamente (número, texto y gold) con una acción del plan, y las acciones sin gold (el 30.6 %) son justo los pasos que faltan. Por eso existe, por ejemplo, una consulta con solo `step3`: los pasos 1 y 2 no tenían gold. Las 90 consultas sin ningún paso son las que no tenían gold en ninguna acción.

> **Interpretación:** los "huecos" no son errores, sino un filtrado deliberado. En test probablemente solo se den los pasos que tienen gold.

---

## 6. Guidance temporal (sección 6)

`guidance_*.jsonl` trae metadatos generados (probablemente por un LLM) sobre la temporalidad de cada consulta y de sus gold. **Seguramente no estará en test**, así que se trata como información para **entender** los datos y, como mucho, para **entrenar**, no para usarla al buscar.

### 6.1 Distribuciones

**Cómo leerlo.** Las tablas dan el porcentaje de consultas por valor. Los mapas de calor muestran, para cada dominio (fila), qué porcentaje de sus consultas cae en cada categoría (columna); cuanto más oscuro, más porcentaje.

**Qué significa.**
- El **95 %** de las consultas está marcada como temporal.
- **Intenciones** principales: `when` (cuándo pasó algo, 23 %), `period_definition` (definir un periodo, 20 %), `timeline` (secuencia de eventos, 20 %), `ongoing_status` (¿sigue siendo así?, 11 %) y `duration` (cuánto dura, 8 %).
- **Clase primaria** dominante: `event_analysis_and_localization` (36 %), seguida de `time_period_contextualization` (21 %).
- **Las etiquetas tienen ruido:** `temporal_intent` contiene a veces nombres de clase (por ejemplo, `causation_analysis`) en vez de intenciones, y `expected_granularity` es **texto libre con 93 valores distintos** (year, decade, "time of day", "real-time"…). En el notebook, los valores con menos de 15 consultas se agrupan en "otras".

> **Interpretación:** si se usa guidance para entrenar (por ejemplo, un clasificador de intención temporal), primero hay que **normalizar las etiquetas**.

### 6.2 `retrieval_plan` vs. `steps`
Repite la conclusión de 5.5: el 30.6 % de las acciones del plan no tienen gold y no se convierten en pasos.

### 6.3 Anotaciones de los gold y coincidencia de anclas

**Qué se hizo.** Para cada par (consulta, documento gold) se extrajeron los años mencionados en las anotaciones de la consulta (`key_time_anchors`, `query_temporal_signals`) y del pasaje (`time_mentions`, `time_scope_guess`…), y se miró si comparten **al menos un año**.

**Cómo leerlo.** `%comparten_año (si ambos tienen)` solo cuenta los pares en que **ambos** lados mencionan algún año. `pares_comparables` dice sobre cuántos pares se calculó; si es pequeño (3, 4…), el porcentaje no es fiable.

**Qué significa.** Solo el **45 %** de los pares comparables comparte algún año. Un gold puede ser relevante sin mencionar el año exacto: puede hablar del periodo anterior, del contexto o de la causa.

> **Interpretación:** usar "el documento debe contener el año de la consulta" como **filtro duro** eliminaría más de la mitad de los gold. Como mucho, puede ser una señal suave para el reranker.

### 6.4 Rendimiento por clase temporal

**Cómo leerlo.** nDCG@10 medio de BM25 por categoría. `n` es el número de consultas, `pct_nDCG_0` el porcentaje de consultas sin ningún gold en el top-10, y `fiable = False` marca los grupos de menos de 15 consultas, cuyas cifras no hay que tomar en serio.

**Qué significa.** Entre los grupos fiables:

| Más difíciles (nDCG@10) | | Más fáciles (nDCG@10) | |
|---|---:|---|---:|
| `materials_artifacts_and_provenance` | 0.063 | `event_analysis_and_localization` | 0.137 |
| `time_period_contextualization` | 0.067 | `event_verification_and_authenticity` | 0.129 |
| intención `period_definition` | 0.060 | intención `when` | 0.140 |
| intención `duration` | 0.067 | intención `order` | 0.163 |

> **Interpretación:** las preguntas sobre un **evento concreto** ("¿cuándo pasó X?") son más fáciles, porque el evento tiene un nombre que aparece en los documentos. Las preguntas sobre **periodos, duraciones o contexto histórico** son más difíciles, porque la respuesta está repartida y no tiene palabras clave claras. Ahí es donde un sistema con razonamiento temporal puede ganar más.

---

## 7. Baselines y diagnóstico de errores (sección 7)

### 7.1 Reproducción del baseline oficial

**Cómo leerlo.** `*_ref` es la cifra oficial, `*_nuestro` la de la reimplementación y `Δ` la diferencia.

**Qué significa.** Todas las diferencias son ≤ 0.0001 (macro 0.1075 en 1a y 0.1024 en 1b). **La reproducción es exacta**, así que tenemos el mismo punto de partida que el resto de participantes.

### 7.2 Recall por profundidad: ¿dónde está el cuello de botella?

**Concepto clave.** Un sistema de búsqueda moderno suele tener dos etapas:
1. **Primera etapa (retriever):** rápida y aproximada. Elige ~100–1000 candidatos de entre todo el corpus.
2. **Segunda etapa (reranker):** lenta y precisa. Reordena solo esos candidatos.

El reranker **no puede encontrar un gold que la primera etapa no haya traído**. Por eso se mira el **recall a distintas profundidades** (ver glosario):

| | R@10 | R@100 | R@1000 |
|---|---:|---:|---:|
| BM25 1a (macro) | 0.140 | 0.336 | 0.640 |
| BM25 1b (macro) | 0.166 | 0.392 | 0.701 |

**Cómo leer el gráfico.** Cada fila es un dominio, con tres puntos: R@10 (naranja), R@100 (verde) y R@1000 (azul). La distancia entre el naranja y el azul es lo que un reranker **podría** ganar como máximo. Lo que falta del azul hasta 1.0 está **perdido** en la primera etapa.

**Qué significa.**
- Incluso mirando 1000 documentos, BM25 solo encuentra el **64 %** de los gold en 1a. En `history`, solo el **33 %**.
- Hay mucho margen entre R@10 y R@1000, así que un reranker ayudaría. Pero el ~35 % de los gold que BM25 no encuentra **nunca** se recuperarán con BM25 como primera etapa.

> **Decisión:** la **primera etapa** es el cuello de botella principal y hay que mejorarla antes de invertir en el reranker.

### 7.3 Baseline denso (BGE-small) y fusión híbrida

**Qué se hizo.** Se probó un modelo de embeddings pequeño y sin ajustar (`bge-small-en-v1.5`, ver glosario "recuperación densa") en los tres dominios más pequeños. También se probó la fusión **RRF** de BM25 + denso.

**Cómo leerlo.** Para cada dominio y sub-tarea se comparan las tres columnas: BM25, denso y RRF.

**Qué significa.**

| 1a | BM25 nDCG@10 | Denso nDCG@10 | BM25 R@1000 | Denso R@1000 |
|---|---:|---:|---:|---:|
| iota | 0.156 | **0.268** | 0.729 | **0.881** |
| law | 0.095 | **0.261** | 0.774 | **0.986** |
| quant | 0.026 | **0.151** | 0.477 | **0.794** |

- **El modelo denso, incluso pequeño y sin entrenar con estos datos, supera ampliamente a BM25**: llega a multiplicar el nDCG@10 por 2.7 en `law` y por 6 en `quant`, y el recall a 1000 sube hasta 0.8–0.99.
- La fusión RRF **solo mejora en `iota`**. En `law` y `quant` queda por debajo del denso solo, porque BM25 aporta más ruido que señal en estos dominios.
- En 1b se ve el mismo patrón.

> **Decisión:** **la primera etapa debe ser densa.** El siguiente paso es probar modelos más grandes (bge-base/large, e5, gte), con chunking. Ojo: son 3 dominios pequeños, así que hay que confirmarlo en los grandes (ver 7.4).

### 7.4 Varianza por dominio

**Qué se hizo.** Con **bootstrap** (ver glosario) se calculó un **intervalo de confianza al 95 %** (IC95) del nDCG@10 de cada dominio. Ese intervalo responde a la pregunta: "si tuviéramos otras consultas parecidas, ¿entre qué valores estaría el resultado?".

**Cómo leer el gráfico.** Cada punto es el nDCG@10 de un dominio y la barra horizontal es su IC95. Los dominios están ordenados por número de consultas, de menos (abajo) a más (arriba). **Barra larga = mucha incertidumbre.**

**Qué significa.**

| Dominio | n consultas | nDCG@10 | IC95 | Ancho |
|---|---:|---:|---|---:|
| iota | 7 | 0.156 | [0.03, 0.30] | 0.27 |
| law | 24 | 0.095 | [0.02, 0.20] | 0.19 |
| history | 561 | 0.065 | [0.05, 0.08] | 0.03 |

- En `iota`, el valor "real" podría estar en cualquier punto entre 0.03 y 0.30. **Una mejora de 0.05 en `iota` puede ser pura suerte.**
- En `history` el intervalo es estrecho y los cambios son fiables, pero pesa solo 1/13 en la nota.

> **Decisión:** no decidir nada mirando un dominio pequeño aislado. Comparar siempre en la macro-media y, cuando las diferencias sean pequeñas, mirar si los intervalos se solapan.

### 7.5 Revisión manual de 20 consultas fallidas

**Qué se hizo.** Se eligieron 20 consultas con nDCG@10 = 0 (ningún gold en el top-10), repartidas entre dominios, y se leyó la consulta, un fragmento del gold y el primer documento recuperado.

**Cómo leerlo.** `mejor_rank_gold` es la posición del primer gold (`inf` = no está en el top-1000) y `jaccard_max` es el mayor solapamiento léxico entre la consulta y sus gold.

**Qué significa.** Aparecieron cinco patrones:

1. **El top-1 suele ser relevante aunque no sea gold.** Por ejemplo, para "Is the 22nd Amendment self-executing?" (law), BM25 trae un texto que cita la Sección 3 de la 14ª enmienda, que la propia consulta compara. Los qrels son **incompletos**: marcan un pasaje de evidencia concreto, no todo lo que es relevante. Parte de los "fallos" no son fallos reales, y no hay que obsesionarse con ellos.
2. **Gold con poca relación aparente.** Por ejemplo, "How long do Zero Value Transactions last?" (iota) tiene como gold un aviso de degradación del servicio EVM de Shimmer. Ningún buscador léxico encuentra esto; requiere conocimiento del dominio.
3. **Documentos basura arriba:** estado JavaScript (travel), tablas de Hacker News (bitcoin), páginas de copyright (bitcoin). Son textos largos con muchas palabras distintas, que acumulan puntos frente a consultas largas.
4. **La temporalidad se ignora:** "Finding 1834 location in St Clement" recupera listas de jueces con fechas. BM25 trata "1834" como una palabra más.
5. **Consultas muy largas** (340 y 539 palabras en dos casos) **diluyen** los términos clave, cuando el título suele resumir bien lo que se busca.

**Composición del top-10.** La última tabla comprueba si los falsos positivos son documentos de ID largo con el mismo número que la consulta (ver 2.4). Solo lo son en el 3.7 % de los casos en total (18 % en `law`, 12.5 % en `politics`). Los falsos positivos son sobre todo documentos normales.

---

## 8. Resumen: qué nos dice todo esto

### Los cinco mensajes clave

1. **Los datos son fiables**, pero hay **un artefacto en los IDs** que delata a los gold. No hay que usarlo.
2. **El problema principal es de vocabulario:** los gold hablan del tema con otras palabras. BM25 pierde el ~35 % de los gold incluso a profundidad 1000.
3. **Los modelos densos resuelven buena parte de eso:** incluso uno pequeño y sin ajustar duplica o triplica el nDCG@10 de BM25 en los dominios probados.
4. **La temporalidad no se resuelve con reglas superficiales** (contar años, filtrar por año). Las consultas sobre periodos y duraciones son las más difíciles.
5. **Los dominios pequeños son muy ruidosos**, así que hay que evaluar con cuidado.

### Plan de pipeline que sugiere el EDA

```
consulta ──► limpieza HTML ──► (opcional) reformulación con LLM / HyDE
                                   │
                                   ▼
                     retriever DENSO sobre chunks  (+ BM25 como señal auxiliar)
                                   │   top-100 / top-1000
                                   ▼
                     reranker cross-encoder entrenado con train
                     (hard negatives del propio corpus)
                                   │
                                   ▼
                     diversificación del top-10 (casi-duplicados)
                                   │
                                   ▼
                                ranking
```

En 1b: buscar con el paso y con la consulta por separado y fusionar los rankings.

### Lo que **no** hay que hacer
- Usar el formato o la numeración de los IDs.
- Usar `gold_answers` o `guidance` al buscar (probablemente no estarán en test).
- Filtrar documentos por "contiene el año de la consulta".
- Decidir mirando `iota`, `quant` o `law` aisladamente, o mirando dev.

### Preguntas para los organizadores
1. ¿Qué campos tendrá el test? ¿Habrá guidance? ¿Se darán las `step_instruction` en 1b?
2. ¿Tendrá el corpus de test la misma mezcla de IDs cortos y largos?
3. ¿Cuál es la profundidad máxima de los rankings en la submission?

---

## 9. Glosario

### Métricas de recuperación

**Ranking / run.** La lista ordenada de documentos que devuelve el sistema para una consulta. Un *run* es el conjunto de rankings de todas las consultas.

**Qrels / gold.** Los juicios de relevancia: qué documentos son relevantes (gold) para cada consulta. Aquí son binarios: relevante (1) o no.

**Recall@k.** De todos los gold de una consulta, ¿qué fracción aparece entre los *k* primeros resultados?
- Ejemplo: 4 gold, y 2 de ellos están en el top-100 → Recall@100 = 2/4 = 0.5.
- Mide la capacidad de **encontrar**, sin importar el orden dentro del top-k.

**nDCG@10 (normalized Discounted Cumulative Gain).** La métrica oficial. Premia tener gold arriba y premia más cuanto más arriba estén:
1. Cada gold en la posición *i* suma `1 / log₂(i + 1)`: vale 1.0 en la posición 1, 0.63 en la 2, 0.5 en la 3… y 0.29 en la 10. Lo que está por debajo de la 10 no cuenta.
2. Esa suma (**DCG**) se divide por la del ranking perfecto (**IDCG**), con todos los gold en las primeras posiciones. Así el resultado va de 0 a 1.
- Ejemplo: 2 gold, el sistema pone uno en la posición 1 y otro en la 4. DCG = 1 + 1/log₂5 = 1 + 0.43 = 1.43. Ideal (posiciones 1 y 2): 1 + 0.63 = 1.63. nDCG = 1.43 / 1.63 = 0.88.
- **Referencia:** un nDCG@10 de 0.1 (el baseline) significa que la mayoría de las consultas no tienen ningún gold en el top-10.

**Macro-media.** Media de las medias por dominio: cada dominio pesa lo mismo, tenga 7 o 561 consultas. Lo contrario sería la *micro-media*, que promedia todas las consultas juntas.

### Modelos de búsqueda

**BM25.** El buscador léxico clásico. Puntúa un documento según las palabras de la consulta que contiene, con tres ideas:
- **tf (term frequency):** cuantas más veces aparece la palabra, mejor, pero con **saturación**: la décima aparición suma mucho menos que la primera (lo controla `k1 = 0.9`).
- **idf (inverse document frequency):** las palabras raras ("halving") pesan mucho más que las comunes ("bitcoin" en el dominio bitcoin).
- **Normalización por longitud:** un documento largo tiene más palabras por casualidad, así que se le penaliza un poco (lo controla `b = 0.4`).
- Limitación: solo encuentra documentos que usan **las mismas palabras** (después de pasarlas a su raíz: "running" → "run"). No sabe que "orphan block" y "stale block" significan lo mismo.

**Analizador de Lucene.** Preprocesa el texto antes de BM25: pasa a minúsculas, quita *stopwords* ("the", "of"…) y reduce cada palabra a su raíz (*stemming* de Porter).

**Recuperación densa / embeddings.** Un modelo neuronal (aquí BGE) convierte cada texto en un vector de 384 números que representa su **significado**. Textos con significado parecido tienen vectores cercanos aunque usen palabras distintas. Buscar es encontrar los vectores de documento más cercanos al de la consulta (similitud coseno). Su limitación: lee como máximo 512 tokens.

**Token.** La unidad que procesa un modelo neuronal: palabras o trozos de palabra ("halving" → "hal", "##ving"). Aquí, ~1.45 tokens por palabra.

**Chunking.** Partir los documentos largos en trozos que el modelo pueda leer enteros. El documento recibe la puntuación de su mejor trozo.

**Reranker (cross-encoder).** Un modelo que lee la consulta y el documento **juntos** y decide su relevancia. Es mucho más preciso que el retriever, pero lento, así que solo se aplica a los ~100 primeros candidatos.

**Hard negatives.** Documentos del mismo tema que la consulta pero que **no** son relevantes. Son los mejores ejemplos negativos para entrenar un modelo, porque le obligan a aprender diferencias finas.

**RRF (Reciprocal Rank Fusion).** Forma simple de combinar varios rankings: cada documento suma `1 / (60 + posición)` por cada ranking en el que aparece. No necesita que las puntuaciones de los sistemas sean comparables.

**HyDE (Hypothetical Document Embeddings).** Un LLM escribe una respuesta hipotética a la consulta y se busca con esa respuesta en vez de con la pregunta. Funciona porque una respuesta se parece más a un documento que una pregunta.

### Estadística y análisis

**Percentil.** El valor por debajo del cual queda cierto porcentaje de los datos. La mediana es el percentil 50 (p50).

**Similitud de Jaccard.** Para dos conjuntos A y B: `|A ∩ B| / |A ∪ B|`, es decir, los elementos comunes divididos por los elementos totales. Va de 0 (nada en común) a 1 (idénticos).
- Ejemplo: consulta {bitcoin, fork, store} y documento {bitcoin, fork, chain, node} → 2 comunes / 5 totales = 0.4.

**MinHash + LSH.** Técnica para encontrar casi-duplicados en millones de documentos sin compararlos todos contra todos:
- **MinHash** resume cada documento en una "firma" de 64 números, de forma que la fracción de números iguales entre dos firmas **estima** su Jaccard.
- **LSH** agrupa las firmas que coinciden en algún bloque, de modo que solo se comparan documentos que probablemente sean parecidos.

**AUC (área bajo la curva ROC).** Mide si un clasificador ordena bien: la probabilidad de que, tomando un gold y un no-gold al azar, el modelo le dé mayor puntuación al gold. 0.5 = azar, 1.0 = perfecto. Orientativamente: 0.6 es débil, 0.7–0.8 aceptable y 0.9 o más muy bueno.

**Validación cruzada (5 folds).** Se divide la muestra en 5 partes: se entrena con 4 y se evalúa con la restante, 5 veces, y se promedia. Así el resultado no depende de una división concreta.

**Log-odds con prior de Dirichlet (z-score).** Mide si una palabra aparece significativamente más en un grupo que en otro, corrigiendo para que las palabras rarísimas no den resultados exagerados. El z-score se lee como "cuántas desviaciones estándar" de diferencia hay: |z| > 2 es significativo.

**Bootstrap e intervalo de confianza (IC95).** Para saber cuánto podría variar una media si tuviéramos otras consultas, se simula: se remuestrean al azar (con reemplazo) las consultas del dominio 2,000 veces y se calcula la media cada vez. El IC95 va del percentil 2.5 al 97.5 de esas medias. Si los IC de dos sistemas se solapan mucho, no podemos asegurar que uno sea mejor.

**Leakage (fuga de información).** Cuando el sistema aprende o usa una pista que delata la respuesta pero que no existiría en un uso real: aquí, el formato del ID. Infla los resultados en desarrollo y no se generaliza al test.

**Expresión regular (regex).** Patrón de búsqueda de texto, por ejemplo "4 dígitos entre 1500 y 2039" para detectar años. Es rápida pero aproximada: también captura cantidades como "1500 USD".

---

## 10. Detalles prácticos del notebook

- **Caché:** todo lo costoso se guarda en `outputs/eda/`:

  | Carpeta / archivo | Contenido |
  |---|---|
  | `corpus_stats/` | rasgos por documento + textos de gold y de una muestra del 1 % |
  | `bm25/` | matrices BM25 de los 13 dominios |
  | `runs/` | rankings de todas las variantes |
  | `minhash_*.npz` | firmas MinHash |
  | `dense/` | embeddings BGE de iota, quant y law |

  La primera ejecución tarda ~30 min; con caché, menos de 1 min. Para recalcular algo, borra su archivo. Ocupa ~1.3 GB, así que conviene añadir `outputs/` al `.gitignore`.
- **Reiniciar el kernel** borra las variables en memoria: hay que ejecutar desde arriba (*Run All*), pero gracias a la caché es rápido.
- **FAISS no se usa:** en este entorno el kernel se caía al combinar faiss/torch con la JVM de Pyserini. La búsqueda densa se hace con numpy, que da el mismo resultado exacto. Para los dominios grandes conviene usar FAISS en un script aparte, sin Pyserini.
- **Si la celda densa (7.3) se queda colgada en VS Code**, ejecuta el notebook desde la terminal:
  ```bash
  cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace eda_track1_tempo.ipynb
  ```
- **Dependencia añadida:** `pytrec-eval-terrier` (la librería de la métrica oficial), instalada con `uv add`.
