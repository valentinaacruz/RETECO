# Revisión de la extracción con reglas de TG-RAG (2026-11-11)

Antes de la primera ejecución de TG-RAG, [`PLAN_TGRAG.md`](PLAN_TGRAG.md) §6 pedía revisar a mano `temporal.extract_spans` con oraciones reales de cada dominio. Este documento recoge esa revisión: qué errores aparecieron, qué se cambió, **por qué** y **dónde está el código**, y cuánto mejora la extracción en una muestra que no se usó para diseñar las reglas.

Todo es preparación de TG0. **No se ha ejecutado ningún experimento de ranking** y no se tocó dev.

**Resumen** (muestra de evaluación, semilla 2, 5,200 documentos; detalle en §4):

| | Antes | Después |
|---|---:|---:|
| Hechos por documento | 12.2 | 11.9 |
| % de hechos con τ | 27.6 % | 31.5 % |
| Entidades distintas | 48,840 | 43,272 (−11 %) |
| Entidades unidas por "and" (% de menciones, rango por dominio) | 3.5–7.6 % | ≤ 0.06 % |
| Entidades de más de 5 palabras (% de menciones) | 2.6–9.3 % | ≤ 0.5 % |
| Menciones cuya entidad aparece en ≥ 2 documentos | 29.3 % | 29.9 % |
| Años de referencias bibliográficas marcados (`citation`) | — | 12.1 % de las expresiones; precisión 40/40; recall ≈ 35–45 % |
| Fechas de la página marcadas (`metadata`) | — | 4.1 % de las expresiones; precisión 40/40 |
| Velocidad del extractor (docs/s, 1 núcleo, M2 Pro) | 952 | 552 |

**Decisión:** los años de cita y las fechas de metadatos **no se quitan por defecto**. Se marcan y `tg_extract.py --drop-time citation metadata` los quita en una carpeta de hechos aparte (`rules-nocit-nometa`), para medir en TG0 si quitarlos ayuda (§3.3, §6).

---

## 1. Cómo se revisó

- **Muestra:** para cada uno de los 13 dominios, 400 documentos al azar de los primeros 6,000 del corpus (`scripts/tg_review_extraction.py`, función `sample_docs`). Cada semilla da documentos distintos.
- **Separación diseño / evaluación**, para no evaluar las reglas sobre los mismos ejemplos con los que se escribieron:
  - **semilla 0:** revisión inicial y primera ronda de reglas;
  - **semilla 1:** segunda ronda (errores que quedaban);
  - **semilla 2:** evaluación final. No se cambió ninguna regla después de mirarla.
- **Precisión:** 40 expresiones marcadas al azar de cada papel, revisadas a mano.
- **Recall de las citas:** 60 años sueltos *sin* marcar al azar, revisados a mano para contar cuántos eran referencias que se escaparon. Es una estimación con mucho margen (con 60 ejemplos, ±12 puntos aprox.).
- **Antes / después:** el mismo script sobre la versión del commit `066fe52` (código anterior) y sobre la actual, con la misma semilla.

Para reproducirlo:

```bash
.venv/bin/python tests/test_extraction.py                                          # 16 pruebas
.venv/bin/python scripts/tg_review_extraction.py --seed 2 --show 40 --show-unflagged 60
```

La columna "antes" sale de ejecutar el mismo script desde una copia del commit `066fe52` (con `data/` enlazada).

## 2. Qué se encontró

Revisión de ~130 expresiones temporales y ~65 grupos de entidades (semilla 0). Las proporciones son estimaciones a ojo sobre esa muestra.

**Expresiones temporales.** Las fechas reales se reconocen bien en todos los formatos ("October 1942", "10 March 1880", "the 1930s", "early 15th century", "1925- 1929"). Los errores eran de tres tipos:

| Tipo | Frecuencia aprox. | Ejemplos |
|---|---|---|
| Años de **citas y referencias**: fecha bien leída, pero no es el momento de un hecho | ~1 de cada 4 | `(Oxford, 1964)`, `Frankel and Lee’s (1998)`, `[2014] UNSWLawJl 39` |
| **Números que no son fechas** | ~6 % | `uid=1738` y `dci-2004/` dentro de URLs, `1808 \|5А) slots`, `1464(t)`, `printed page 1654`, `$1200-1500` como rango |
| **Fechas de la página** (metadatos) | ~5 % | `August 4, 2025 at 7:09 pm` (comentarios de foro), `(accessed 2 May 2013)`, `Date: April 28, 2023` |

**Entidades.** Eran la parte débil:

- palabras de inicio de oración tomadas como nombres: ANOTHER, FINALLY, MOREOVER, THANKS, PRIOR, LIKE, ONCE, TWO, CURRENT, FIXED…;
- nombres distintos fusionados por "and" (`TYLER AND CAMERON WINKLEVOSS`, `YANG AND MILLS`) o por puntuación (`MCKEE AMERICAN ELECTRIC RAILWAY ASSOCIATION EVA B`, `SAN FRANCISCO CALIFORNIA`);
- basura de texto que no es prosa: código (`X10 MOV STACKPREEMPT X11 BEQ X10 X11`), spam con `[url=…]`, titulares en mayúsculas.

Una entidad fusionada casi nunca coincide entre documentos, así que el grafo queda poco conectado y el PPR tiene poco por donde propagar (riesgo ya anotado en PLAN_TGRAG §7).

**Cobertura temporal por dominio:** 20–27 % de las oraciones tienen alguna fecha en `genealogy`, `history` y `politics`; 6–8 % en los técnicos (`bitcoin`, `cardano`, `iota`, `monero`, `quant`, `workplace`). Coincide con la hipótesis H1: si TG-RAG aporta, será sobre todo en los primeros.

## 3. Cambios

### 3.1 No buscar fechas dentro de URLs

- **Qué:** antes de aplicar los patrones, las URLs, correos, dominios (`example.com/…`) y etiquetas BBCode (`[url=…]`, `[b]`) se marcan como ya usados y ningún patrón puede casar dentro.
- **Por qué:** los números de una URL (`uid=1738`, `renault-trafic-dci-2004/`) no son fechas del contenido.
- **Dónde:** [`src/reteco/temporal.py`](src/reteco/temporal.py), `URL_RE` (línea 219) y su uso al principio de `extract_spans_with_role` (línea 404). `URL_RE` también la usa `tgraph` para ignorar URLs al buscar entidades.
- **Prueba:** `test_no_dates_inside_urls`.

### 3.2 Números que no son años

- **Qué:**
  1. `_NOT_YEAR_BEFORE` (línea 211): un año suelto o un rango no cuenta si lo precede, en los 25 caracteres anteriores, una palabra de página, sección o identificador (`page`, `section`, `form`, `issue`, `id`, `uid`, `isbn`…), una abreviatura **con punto** (`p.`, `pp.`, `no.`, `vol.`, `art.`, `fig.`…), un símbolo (`§`, `#`, `=`, `$`, `£`, `€`) o una página de repertorio judicial (`F.2d 1284, 1307-08`).
  2. `_NOT_YEAR_AFTER` (línea 202): se amplió la lista de unidades (`coins`, `tokens`, `shares`, `hours`, `students`…) y se añadieron los subapartados pegados al número (`1464(t)`). Ahora se aplica también a los rangos.
  3. Un número descartado **consume su tramo** (centinela `_NOT_A_DATE`, línea 248, aplicado en `_make_span`, línea 251). Sin esto, al descartar el rango `$1200-1500`, el patrón de año suelto se quedaba con `1500`.
  4. El patrón `range` (línea 241) ya no excluye `$` en su lookbehind, para poder casar el importe y descartarlo; el de año (línea 244) excluye además `=` y `@`.
- **Por qué las abreviaturas exigen punto:** sin punto, palabras como "art" o "no" quitarían años reales ("Soviet art 1920").
- **Prueba:** `test_numbers_with_page_section_money_context`; `test_real_dates_still_found` comprueba que "Soviet art 1920" o "mid-2021" siguen siendo años.

### 3.3 Papel de la fecha: `citation` y `metadata` (opcional)

- **Qué:** cada expresión temporal recibe un papel:
  - `citation`: año de una referencia bibliográfica o judicial;
  - `metadata`: fecha de la página y no del contenido;
  - `None`: fecha del contenido.

  `extract_spans(text, drop=("citation", "metadata"))` las descarta; **por defecto no descarta nada**.
- **Por qué es opcional y no un borrado:** un año de cita puede seguir dando pistas del periodo del documento (un texto que cita libros de 1964 sobre política de 1820), y el paper no hace este filtrado. Si ayuda o no se decide con datos en TG0, siguiendo la regla de ajustar en train.
- **Dónde:** [`src/reteco/temporal.py`](src/reteco/temporal.py):
  - `ROLES` (línea 313) y `time_role` (línea 362): las reglas;
  - `extract_spans_with_role` (línea 404): devuelve `(inicio, fin, TimeSpan, papel)`;
  - `extract_spans` (línea 431): parámetro `drop`. Un papel desconocido da `ValueError`. Si `drop` está vacío, no se calcula el papel.
- **Reglas de `metadata`** (todas las granularidades):
  - precedida de `accessed`, `retrieved`, `last updated`, `posted on`, `Date:`, `Published:`, `©`, `copyright`… (`_META_BEFORE`, línea 314);
  - o seguida de una hora (`at 7:09 pm`, `, 3:30 PM`) (`_META_AFTER`, línea 320).
- **Reglas de `citation`** (solo años sueltos: en las referencias las fechas completas son raras y ambiguas):

  | Señal | Regex (línea) | Ejemplo |
  |---|---|---|
  | `et al.` antes del año | `_ETAL_BEFORE` (321) | `Wang et al., 2016` |
  | "Apellido, I." antes | `_REFLIST_BEFORE` (324) | `Poterba, J.M. & Venti, S.F., 1992` |
  | año + página, volumen o editor | `_PAGE_AFTER` (353), `_REF_AFTER` (346) | `Long 2005, p.`, `(2015) p. 12`, `2016 edited by`, `2010; 17:1318-1325` |
  | edición o número de volúmenes antes | `_EDITION_BEFORE` (354) | `2nd ed, 2022`, `4 vols, 1516` |
  | pie de imprenta sin paréntesis | `_IMPRINT_BEFORE` (339) | `Toronto: University of Toronto Press, 2011` |
  | dentro de un paréntesis o corchete que se cierra tras el año, con autor(es), editorial o tribunal delante | `_AUTHOR_IN_PAREN` (330), `_PUBLISHER_IN_PAREN` (334), `_COURT_IN_PAREN` (349), `_CLOSE_AFTER` (351) | `(see Feferman 1960)`, `(Blinder, Goodhart, & Wyplosz, 2001)`, `(London: Z ed, 1978)`, `(Dover Pub., 1956)`, `(1st Dept 2003)` |
  | `(año)` solo, tras posesivo, `et al.`, inicio de fragmento, páginas de revista o "Apellido, Nombre" | dentro de `time_role`; `_PAGES_BEFORE_PAREN` (359), `_REFENTRY_BEFORE_PAREN` (343) | `Lee’s (1998)`, `(2017).`, `25(3), 505–512 (1995)`, `Peacock, Darren (2002)` |
  | `[año]` seguido de un repertorio | dentro de `time_role` | `[2014] UNSWLawJl 39` |
  | paréntesis abierto en el fragmento anterior | `_ORPHAN_CLOSE`, `_AUTHOR_BEFORE` (357–358) | `Boutroux 1908).` (el divisor de oraciones corta en `pp. 33`) |

- **Excepciones**, para no marcar hechos como citas:
  - los **años de titulación** (`(MBBCh 1965)`, `D.Sc., 1949`) no son citas: son hechos biográficos que importan en `genealogy` (`_DEGREE_BEFORE`, línea 326; en la semilla 0 eran los 3 únicos errores de 40);
  - las iniciales de `_REFLIST_BEFORE` tienen que ir seguidas de espacio, coma u otra inicial, para que `England, D.Sc.` no parezca "Apellido, I.";
  - `Apollo 11 (1969)` o `Homestead (2016)` no son citas: un nombre seguido de `(año)` es muchas veces un hecho;
  - una etiqueta (`Note:`, `Source:`…) no cuenta como ciudad de un pie de imprenta.
- **En la línea de comandos:** [`scripts/tg_extract.py`](scripts/tg_extract.py), opción `--drop-time` (líneas 68–75) y paso a `extract_rules` (línea 94). Cada combinación va a su propia carpeta de hechos (`rules-nocit`, `rules-nometa`, `rules-nocit-nometa`, `rules-inherit-nocit`…; sufijos en `_ROLE_SUFFIX`, línea 38). Así las variantes no se mezclan en la caché incremental, y ese nombre es el que se pasa a `--extractor` en `tg_diagnose.py` y `tg_retrieve.py`.
- **En la extracción de hechos:** `extract_rules(..., drop_time=...)` en [`src/reteco/tgraph.py`](src/reteco/tgraph.py) (línea 220).
- **Pruebas:** `test_citation_years`, `test_event_years_are_not_citations`, `test_metadata_dates`, `test_drop_roles`.

### 3.4 Entidades y filtro de prosa

Todo en [`src/reteco/tgraph.py`](src/reteco/tgraph.py):

| Cambio | Por qué | Dónde (línea) |
|---|---|---|
| **Palabras de discurso** (`another`, `finally`, `moreover`, `thanks`, `prior`, `like`, `once`, `two`…) y **tratamientos** (`mr`, `dr`, `prof`…) añadidos a la lista de exclusión | aparecían como entidades al abrir oraciones | `_DISCOURSE` (76), `_TITLES` (74), `_STOP_CAPS` (89) |
| **Primera palabra de la oración**: si es una entidad de una sola palabra, solo se acepta si esa palabra aparece capitalizada en mitad de otra oración del mismo chunk, o es un acrónimo | una lista nunca cubre todas las palabras de inicio ("Monetization is…", "Platforms like…"), pero un nombre real suele repetirse en mitad de oración ("Monero is…" + "…users of Monero…") | `mid_sentence_caps` (139), parámetro `known` de `rule_entities` (163); `extract_rules` calcula `known` por chunk |
| **"and", "&" e "y" ya no unen nombres** | `TYLER AND CAMERON WINKLEVOSS` debe ser dos entidades; se pierden nombres como "Procter & Gamble", que es un coste menor | `_CONNECTORS` (72) |
| **La puntuación corta la entidad** (comas, punto y coma, paréntesis, comillas), salvo el punto de una inicial ("John F. Kennedy") | `SAN FRANCISCO, CALIFORNIA` → dos entidades; acaba con las fusiones de referencias bibliográficas | `_gap_breaks` (155), usada en `rule_entities` |
| **Acrónimos con puntos** como un token y normalizados sin puntos ("U.S." → `US`, "U.S. Army" → `US ARMY`) | antes se partían en letras sueltas que se descartaban | `_TOKEN_RE` (69), `normalize_entity` (102) |
| **Se descartan entidades de más de 5 palabras** (sin contar conectores) | casi siempre son titulares o referencias fusionadas | `MAX_ENTITY_TOKENS` (34) |
| **Filtro de prosa:** no dan hechos las oraciones con BBCode (spam), con marcas de código (`//`, `==`, `=>`, `{`…, o > 10 % de símbolos) ni las de ≥ 20 letras con > 60 % en mayúsculas | ahí "palabra capitalizada = nombre" no funciona y las fechas son basura (`[url=…]`, ensamblador) | `is_prose` (119), `_BBCODE_RE` (113), `_CODE_RE` (115), `_CODE_CHARS` (116); se aplica en `extract_rules` |

Detalles del filtro de prosa:

- **Antes de buscar marcas de código se quitan las URLs**; si no, cualquier `http://` contaba como el comentario `//`.
- **Un `;` final no se considera código**: las enumeraciones legales terminan así ("…shall report;").

`rule_entities(sentence)` sin `known` se comporta como antes respecto a la primera palabra; así se puede comparar con la versión anterior.

**Pruebas:** `test_discourse_words_are_not_entities`, `test_and_does_not_join_names`, `test_punctuation_breaks_entities`, `test_initials_and_acronyms`, `test_long_capitalized_runs_dropped`, `test_sentence_initial_word_needs_support`, `test_is_prose`, `test_extract_rules_end_to_end`.

### 3.5 Herramientas nuevas

| Archivo | Para qué |
|---|---|
| [`tests/test_extraction.py`](tests/test_extraction.py) | 16 pruebas de regresión con los casos reales de esta revisión (los errores corregidos y fechas correctas que no deben romperse). No necesita pytest: `.venv/bin/python tests/test_extraction.py` |
| [`scripts/tg_review_extraction.py`](scripts/tg_review_extraction.py) | revisión sobre una muestra (solo lectura): tabla por dominio con `--seed`, ejemplos de citas y metadatos (`--show`) y años sin marcar (`--show-unflagged`). Sirve también para revisar en el futuro el extractor LLM o nuevas reglas |

## 4. Resultados (semilla 2, no usada para diseñar las reglas)

Por dominio (antes → después):

| Dominio | Hechos/doc | % hechos con τ | % menciones en ≥ 2 docs | % cita | % meta |
|---|---|---|---|---:|---:|
| bitcoin | 8.2 → 6.1 | 19.0 → 17.5 | 25.5 → 24.8 | 12.3 | 5.5 |
| cardano | 7.7 → 5.9 | 13.6 → 9.2 | 25.2 → 24.6 | 2.1 | 1.4 |
| economics | 13.3 → 14.3 | 41.8 → 47.5 | 26.9 → 27.4 | 11.5 | 6.7 |
| genealogy | 19.1 → 21.9 | 43.1 → 47.8 | 23.7 → 27.4 | 6.4 | 0.3 |
| history | 24.0 → 25.7 | 30.1 → 32.2 | 27.6 → 31.3 | 18.1 | 4.4 |
| hsm | 13.0 → 13.7 | 29.9 → 37.1 | 21.3 → 22.0 | 23.7 | 1.6 |
| iota | 10.5 → 10.3 | 12.1 → 15.3 | 40.4 → 35.7 | 14.7 | 7.1 |
| law | 10.8 → 9.7 | 27.1 → 29.7 | 35.6 → 34.4 | 8.3 | 0.4 |
| monero | 10.2 → 9.4 | 16.4 → 19.5 | 29.1 → 27.8 | 4.9 | 4.3 |
| politics | 15.3 → 15.6 | 32.5 → 34.3 | 40.1 → 43.5 | 4.8 | 9.6 |
| quant | 8.9 → 7.7 | 16.3 → 17.1 | 27.2 → 22.3 | 9.8 | 0.0 |
| travel | 9.7 → 7.7 | 24.7 → 24.6 | 27.7 → 27.7 | 16.3 | 0.6 |
| workplace | 7.7 → 7.0 | 19.4 → 24.4 | 34.5 → 30.4 | 9.3 | 10.2 |
| **total** | **12.2 → 11.9** | **27.6 → 31.5** | **29.3 → 29.9** | **12.1** | **4.1** |

("% cita" y "% meta" son la fracción de expresiones temporales con ese papel; antes no existía la marca).

Lectura:

- **Menos hechos en los dominios técnicos**, con más spam y código (`bitcoin`, `cardano`, `travel`, `quant`): el filtro de prosa quita oraciones basura, que casi nunca tenían fecha.
- **Más hechos en `genealogy`, `history` y `economics`:** al partir nombres fusionados, una oración da más pares de entidades.
- **Más hechos con τ en casi todos los dominios** (27.6 → 31.5 %): lo que se quita es sobre todo texto sin fecha. En `cardano` baja porque una parte de sus fechas estaba dentro de spam.
- **La conectividad del grafo apenas mejora** (29.3 → 29.9 % de menciones con la entidad en ≥ 2 documentos). Sube en `genealogy`, `history` y `politics`, los dominios con más señal temporal, y baja en algunos técnicos: allí las entidades que "conectaban" eran en parte palabras de discurso ("FINALLY" sale en muchos documentos). Las reglas no van a dar un grafo denso; el LLM (TG7) sigue siendo la vía para eso.

**Precisión y recall de los papeles:**

| Papel | Precisión (revisión manual) | Recall |
|---|---|---|
| `metadata` | 40/40 (semillas 1 y 2) | no medido (los patrones de página son muy regulares) |
| `citation` | semilla 0: 37/40 (3 años de titulación, ya corregidos) · semilla 1: 40/40 (uno era un número de ticket `[AIRFLOW-1296]`, que tampoco es fecha) · **semilla 2: 40/40** | semilla 0: ≈ 20–25 % · semilla 1: ≈ 35 % · **semilla 2: ≈ 35–45 %** (16 referencias claras y 8 dudosas entre 60 años sin marcar) |

Con 40/40, la precisión es ≥ 92 % con un 95 % de confianza (regla del tres).

**El detector de citas es preciso pero se deja más de la mitad.** Si en TG0 quitar las citas ayuda, el efecto real del filtrado estará infraestimado.

## 5. Límites conocidos y decisiones descartadas

- **Listas autor-año sin paréntesis** (`Simmel, 1900; Crump, 1978`): se probó una regla y **se descartó**. Tienen la misma forma que las carreras de `genealogy` (`Lieutenant, 1904; Commander, 1916; Captain, 1920`), que son hechos biográficos. Hay una prueba que lo impide (`test_event_years_are_not_citations`).
- **Citas que se escapan** (semilla 2). No se añadieron más reglas para no ajustar sobre la muestra de evaluación:
  - listas de tesis o referencias con segundo nombre (`Maliro, Charles Edison (1993)`);
  - autores unidos por `&` sin paréntesis (`Erik Snowberg & Justin Wolfers & Eric Zitzewitz, 2012.`);
  - autores con iniciales detrás (`Hussien HM, Yasin SM (2019)`);
  - repertorios judiciales (`447 (2002).`) y nombres de casos (`Yoder (1972)`);
  - listas de años de un autor (`Weitzman (1998, 2001, 2013)`).
- **El divisor de oraciones corta referencias** (`pp. 33` termina una "oración" porque le sigue un dígito), así que a veces el paréntesis queda en otro fragmento. `_ORPHAN_CLOSE` recupera parte. Se podría arreglar el divisor (`_SENT_RE`), pero cambia todos los hechos y no se tocó en esta revisión.
- **Números que siguen pasando como años:** `Form 1040 or 1040-SR` (la palabra "Form" queda lejos del segundo número), tablas numéricas, listas de contraseñas en spam y benchmarks (`Single Core 1053`). Son poco frecuentes.
- **Rangos de días** (`December 6-7, 2021`) dan solo el año, porque `dmy`/`mdy` no admiten un rango de días. Ya pasaba antes.
- **Entidades:**
  - los términos técnicos en minúscula ("segwit", "census") siguen sin capturarse (PLAN_TGRAG §7);
  - la regla de la primera palabra pierde un nombre que solo aparece al inicio de oraciones en todo el chunk;
  - quitar "and" separa nombres como "Procter & Gamble".
- **Velocidad:** el extractor pasa de ~950 a ~550 documentos/s en un núcleo (M2 Pro), sobre todo por el filtro de prosa y el enmascarado de URLs. Para los ~100 k documentos de los pools son ~3 min por variante, aceptable.

## 6. Qué cambia para TG0

- **No hay que regenerar nada:** todavía no existían hechos extraídos (`outputs/exp/tgrag/` no existe), así que no hay caché antigua que invalidar.
- **Propuesta para TG0:** extraer dos variantes y comparar la cobertura de los gold en `T_q`:

  ```bash
  .venv/bin/python scripts/tg_extract.py --extractor rules --pools 1a/p100
  .venv/bin/python scripts/tg_extract.py --extractor rules --pools 1a/p100 --drop-time citation metadata   # → rules-nocit-nometa
  .venv/bin/python scripts/tg_diagnose.py --subtrack 1a --pool p100 --extractor rules --qtime rules
  .venv/bin/python scripts/tg_diagnose.py --subtrack 1a --pool p100 --extractor rules-nocit-nometa --qtime rules
  ```

- **En el servidor:** los cambios están en el repositorio local; hay que hacer commit y push, y luego `git pull` en `myserver`, antes de lanzar TG0 allí.

## 7. Archivos tocados

| Archivo | Cambio |
|---|---|
| [`src/reteco/temporal.py`](src/reteco/temporal.py) | `URL_RE`; `_NOT_YEAR_BEFORE` y `_NOT_YEAR_AFTER` ampliados; centinela `_NOT_A_DATE`; papeles `citation` / `metadata` (`time_role`, `extract_spans_with_role`); parámetro `drop` en `extract_spans` |
| [`src/reteco/tgraph.py`](src/reteco/tgraph.py) | `is_prose`, `mid_sentence_caps`, `_gap_breaks`; nuevas reglas en `rule_entities` (parámetro `known`); `normalize_entity` con acrónimos; `drop_time` en `extract_rules`; `MAX_ENTITY_TOKENS` |
| [`scripts/tg_extract.py`](scripts/tg_extract.py) | opción `--drop-time` y nombre de carpeta por variante |
| [`scripts/tg_review_extraction.py`](scripts/tg_review_extraction.py) | nuevo: revisión sobre una muestra |
| [`tests/test_extraction.py`](tests/test_extraction.py) | nuevo: 16 pruebas de regresión |
| [`PLAN_TGRAG.md`](PLAN_TGRAG.md) | §3.1, §5 y §6 enlazan a esta revisión |
