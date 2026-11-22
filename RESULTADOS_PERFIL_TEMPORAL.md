# Resultados del perfil temporal implícito (TT0) — estado al 2026-11-21

**Idea de partida.** Aislar las palabras que indican temporalidad, tanto en la consulta como en los documentos (sobre todo en los gold), separando el texto en n-gramas y usando cadenas de Markov o algo tipo el PageRank sobre un grafo temporal del paper de TG-RAG ([`papers/2510.13590v1.pdf`](papers/2510.13590v1.pdf)), para que la recuperación sea más específica.

**Cómo se aterrizó.** Es, casi punto por punto, el *document focus time* de Jatowt, Au Yeung y Tanaka (§0):
- se aprende del propio corpus con qué periodos co-ocurre cada n-grama;
- con eso se le asigna un **perfil temporal** a cualquier texto, aunque no traiga ningún año ("Henry VIII" → 1530s, "Etruscans" → siglo VII a. C.).

Aquí se adapta como diagnóstico (TT0): ¿la similitud entre el perfil de la consulta y el del documento separa gold de no-gold dentro del top-100 del híbrido BM25 + denso? ¿Aporta algo al fusionarla con él?

Las reglas son las de siempre: todo en train, dev sin tocar. Los qrels solo se usan para medir; el léxico se aprende sin gold, solo con el corpus del dominio (sin corpus externos, como piden las reglas de RETECO).

**Estado:**

| Paso | Estado |
|---|---|
| TT0 · separación gold / no-gold y fusión en `politics`, `history` y `hsm` | ✅ §2: señal débil (AUC ≈ 0.6) solo en los dominios históricos; la fusión gana +0.007 a +0.011 en ellos, pero elegida en muestra y sin trasladarse entre subtracks; no se adopta |
| Variantes de grafo del paper (`A_con`, TextRank) | ❌ no se hacen (§4) |
| Error del extractor encontrado de paso ("twenty first century") | ✅ corregido en `reteco.temporal` (§5) |

---

## 0. Referencia

> Adam Jatowt, Ching-Man Au Yeung y Katsumi Tanaka. 2013. **Estimating Document Focus Time.** En *Proceedings of the 22nd ACM International Conference on Information and Knowledge Management (CIKM '13)*, San Francisco, CA, EE. UU., 27 oct.–1 nov. 2013, pp. 2273–2278. ACM. DOI: [10.1145/2505515.2505655](https://doi.org/10.1145/2505515.2505655). PDF: [`papers/jatowt2013_focus_time.pdf`](papers/jatowt2013_focus_time.pdf).

Versión extendida: A. Jatowt, C.-M. Au Yeung y K. Tanaka. 2015. *Generic method for detecting focus time of documents.* Information Processing & Management 51(6), pp. 851–868 (no se usó aquí).

**Qué propone el paper.** El *focus time* de un documento es el periodo del que habla su contenido, que no tiene por qué coincidir con su fecha de creación. Para estimarlo:

1. **Asociación palabra–año** (§3.1), sobre una colección de noticias de 1990–2010 (535 k artículos) con los años 1900–2013:
   - *directa*: Jaccard entre las oraciones que contienen la palabra y las que contienen el año, `A_dir(w, t) = c(w, t) / (c(w) + c(t) − c(w, t))`;
   - *por contexto*: `A_con(w_i, t) = 1/|V| · Σ_j A_dir(w_j, w_i)² · A_dir(w_j, t)`. Un paso más por el grafo de co-ocurrencias: una palabra se asocia a un año si sus vecinas fuertes lo están. Es la parte "de grafo" del método.
2. **Peso temporal de cada palabra** (§3.2): cuánto discrimina el tiempo. Una palabra útil apunta a pocos años: se mide con la **entropía** (`ω^temE = max_j E_j − E_w`) o la **curtosis** de su distribución sobre los años.
3. **Asociación documento–año** (§3.3.1): media de las asociaciones de sus palabras pesadas por el peso temporal (`S_U`), también por frecuencia (`S_TF`) o por importancia **TextRank** (`S_TR`: PageRank sobre el grafo de palabras del documento). La versión extendida `S_EXT` suma una estimación de densidad (KDE gaussiana) de las fechas explícitas del texto.
4. **Focus time** = el año con mayor asociación (`argmax_t S(d, t)`).

**Resultados del paper** (error medio en años; tabla 3):

| Datos | Azar | Solo fechas explícitas | Solo palabras (Prop) | Palabras + fechas (PropExt) |
|---|---:|---:|---:|---:|
| Wikipedia (14.5 fechas/doc.) | 36.5 | 3.02 | 18.3 | **2.83** |
| Libros (4.5 fechas/doc.) | 39.3 | 48.1 | 23.5 | **20.4** |
| Web (2.4 fechas/doc.) | 40.5 | 53.4 | 23.6 | **20.7** |

Dos cosas que importan para TEMPO:
- **Las palabras sirven cuando faltan fechas**, pero con un error de ~20 años: fechan "la época", no la década.
- **Si el texto ya trae muchas fechas, las palabras apenas añaden** (Wikipedia: 3.02 → 2.83, diferencia no significativa).

En la discusión (§5) los autores proponen justo el uso de aquí: comparar la representación temporal de la consulta con la del documento, como complemento de la similitud de contenido.

## 1. Qué se implementó y en qué difiere del paper

Script: [`scripts/temporal_profile_diag.py`](scripts/temporal_profile_diag.py). Un léxico por dominio, construido con todo el corpus del dominio (14 procesos; 0.5–3 min por dominio en `myserver`).

| Pieza | Paper | TT0 |
|---|---|---|
| Colección para aprender las asociaciones | noticias externas (535 k) | el corpus del propio dominio (las reglas prohíben corpus externos) |
| Unidad de co-ocurrencia | oración | oración |
| Fechas | años 1900–2013 con regex | `reteco.temporal.extract_spans` (años, meses, décadas, siglos, rangos, a. C.) |
| Tiempo | años | **bins**: décadas de 1000 a 2039 y siglos de −1000 a 999 (124 bins); TEMPO va de la Antigüedad a 2025 |
| Términos | palabras | unigramas y bigramas sin funcionales ni números (se conservan "before", "during", "since"…) |
| Asociación | Jaccard `A_dir` y de contexto `A_con` | `P(bin | n-grama)` con suavizado de Dirichlet hacia el fondo (α = 5); n-gramas con ≥ 10 oraciones fechadas. Es la matriz de transición de **un paso** de una cadena de Markov n-grama → bin, estimada contando co-ocurrencias. **Sin `A_con`** |
| Peso temporal | entropía o curtosis | **especificidad** = `KL(P(·|w) ‖ P0)`, una entropía relativa al fondo del dominio |
| Perfil de un texto | `S_U`, `S_TF` o `S_TR` (TextRank) | media de `P(·|w)` de sus **20 n-gramas más específicos** (especificidad ≥ 0.3), pesada por la especificidad. **Sin TextRank** |
| Fechas explícitas | KDE gaussiana sumada (`S_EXT`) | distribución de las fechas del texto; suavizado con los bins vecinos (núcleo 0.25 / 0.5 / 0.25) |
| Uso | un año (`argmax`) | el **perfil completo**: coseno entre consulta y documento, tras restar el fondo (si no, todo se parece por concentrarse en los siglos XIX–XXI) |

Tres perfiles:
- **E**: solo fechas explícitas (las que saca el extractor con regex).
- **I**: solo implícito (palabras).
- **EI**: mitad y mitad, el análogo de `PropExt`.

Textos: en 1a, la consulta `limpia`; en 1b, la instrucción del paso más la consulta `limpia`; en los documentos, las primeras 1,500 palabras.

**Medición**, dentro del pool top-100 del híbrido BM25 + denso (RRF, BM25 con peso 0.5; [`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) §2.8):
- **AUC temporal**: para cada consulta con gold y no-gold en el pool, la probabilidad de que un gold tenga más similitud que un no-gold. Se promedia entre consultas; 0.5 es azar. Se informa también en las consultas **sin ninguna fecha explícita**, donde solo actúa el perfil implícito.
- **AUC del híbrido**: lo mismo con el score del propio híbrido, como referencia de cuánto separa ya la base dentro del pool.
- **Fusión**: RRF del orden del híbrido con el orden por similitud temporal, con peso 0.1, 0.25 o 0.5 para el temporal. Las consultas sin perfil conservan el orden del híbrido.

## 2. Resultados (train, 2026-11-21)

Se corrió en tres dominios de contenido histórico con muchas consultas fechadas: `history` y `hsm`, los más históricos, y `politics`, con un 60 % de consultas con fecha ([`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) §2.9). Si no hay señal aquí, no la habrá en el resto.

### 2.1 El léxico aprendido

| Dominio | Oraciones | Fechadas | n-gramas en el léxico (≥ 10 oraciones fechadas) |
|---|---:|---:|---:|
| politics | 1.38 M | 21.8 % | 58,442 |
| history | 3.24 M | 20.2 % | 114,485 |
| hsm | 1.87 M | 18.0 % | 63,633 |

**Palabras ancladas a un periodo (alta especificidad).** Salen las que se esperaban, aunque el año no esté en la oración donde se usan:

| Dominio | n-grama → periodo con más masa |
|---|---|
| politics | "henry viii" → 1530s · "hapsburg" → 1510s · "montfort" → 1260s · "edward iii" → 1340s · "glorious revolution" → 1680s · "andrew jackson's" → 1820s · "philadelphia convention" → 1780s · "marx german" → 1840s |
| history | "shalmaneser", "jehu" → siglo IX a. C. · "qin dynasty" → siglo III a. C. · "aemilius paullus" → siglo II a. C. · "american colonists" → 1700s |
| hsm | "etruscans" → siglo VII a. C. · "plato bce" → siglo IV a. C. · "vigiles" → siglo I |

**Pero el top está muy contaminado:**
- **Las propias expresiones temporales**: "ninth century", "bce", "seventeenth century". Son redundantes con el perfil explícito.
- **Páginas repetidas o plantillas**:
  - en `history`, un catálogo de juegos de rol aporta una veintena de bigramas ("solo game", "retroclones", "asgardian"), todos en "1000s" y con 28 oraciones cada uno;
  - una plantilla de cita genealógica ("beall eds", "ancestral roots", "lee sheppard") cae en 1700s;
  - en `politics`, un aviso legal ("judicial notice", "official editions") cae en 1500s.
- **Números que no son años**: "art nafta" → 1110s (artículo 1110 del TLCAN).
- **Siglos sin era**: "athenian" sale en el siglo IV *d. C.*, porque los textos sobre Atenas dicen "the fourth century" sin "BC" y el extractor lo lee como era común.

**Marcadores temporales (fechabilidad).** Los n-gramas que más aparecen en oraciones fechadas no son "before", "during" ni "era". Son **metadatos y bibliografía**: "accessed june", "archived november", "reuters october", "retrieved cite", "isbn archived", editoriales ("paris grasset", "paris plon"), "died july". Coincide con O5 ([`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) §2.5): el vocabulario más "temporal" del corpus es el de las páginas bibliográficas, que son justo las que se parecen a los hard negatives.

### 2.2 Separación gold / no-gold y fusión con el híbrido

"Consultas" son las consultas (1a) o pasos (1b) con qrels. Las AUC son medias por consulta dentro del pool top-100 (0.5 = azar). "AUC I sin fecha" es el perfil implícito en las consultas sin ninguna fecha explícita, donde el perfil explícito no existe. "Mejor fusión" es el mejor nDCG@10 de 9 combinaciones (perfil E/I/EI × w 0.1/0.25/0.5), elegido en el propio dominio, así que es optimista.

| Dominio | Sub | Consultas | Con fecha explícita | AUC E | AUC I | AUC EI | AUC I sin fecha (n) | AUC híbrido | nDCG@10 base | Mejor fusión |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| politics | 1a | 105 | 61 % | 0.491 | 0.530 | 0.483 | 0.529 (36) | 0.83–0.85 | 0.3952 | 0.3945 (E, w 0.1; −0.001) |
| politics | 1b | 266 | 60 % | 0.524 | 0.571 | 0.525 | 0.562 (93) | 0.89 | 0.4067 | 0.4019 (I, w 0.1; −0.005) |
| history | 1a | 561 | 53 % | 0.631 | 0.620 | 0.622 | 0.610 (245) | 0.77–0.78 | 0.2323 | 0.2420 (E, w 0.5; **+0.010**) |
| history | 1b | 1,421 | 53 % | 0.645 | 0.636 | 0.640 | **0.635** (565) | 0.82 | 0.2499 | 0.2608 (I, w 0.1; **+0.011**) |
| hsm | 1a | 105 | 57 % | 0.636 | 0.601 | 0.593 | 0.553 (41) | 0.82–0.85 | 0.3347 | 0.3435 (E, w 0.1; +0.009) |
| hsm | 1b | 254 | 56 % | 0.635 | 0.603 | 0.596 | 0.565 (92) | 0.82 | 0.3445 | 0.3511 (EI, w 0.1; +0.007) |

La AUC del híbrido se da como rango cuando cambia según el perfil, porque cada perfil se mide en las consultas que tienen ese perfil.

Con w = 0.5 la fusión empeora en 17 de las 18 combinaciones (hasta −0.11 en `politics` 1a); la excepción es `history` 1a con el perfil E.

Similitud media gold / no-gold (perfil I): `politics` 0.37 / 0.30 (1a) y 0.41 / 0.31 (1b); `history` 0.73 / 0.64 y 0.75 / 0.66; `hsm` 0.62 / 0.53 y 0.64 / 0.58.

**¿Se sostiene la mejor fusión fuera de la muestra?** Una comprobación sencilla: elegir la combinación en un subtrack y aplicarla en el otro (comparten consultas, así que es una prueba optimista):

| Dominio | Elegida en 1a → aplicada en 1b | Elegida en 1b → aplicada en 1a |
|---|---|---|
| history | E, w 0.5 → 0.2473 (−0.003) | I, w 0.1 → 0.2368 (+0.005) |
| hsm | E, w 0.1 → 0.3437 (−0.001) | EI, w 0.1 → 0.3329 (−0.002) |

Ninguna combinación fija mejora los tres dominios a la vez: con E y w 0.1, por ejemplo, `history` sube (+0.007 / +0.005) pero `politics` baja (−0.001 / −0.005) y `hsm` 1b también (−0.001).

## 3. Lecturas

- **La intuición es correcta en parte: el periodo se puede inferir sin años.** En `history` 1b, el perfil implícito separa gold de no-gold con AUC 0.64 en 565 pasos sin ninguna fecha explícita. Ni el extractor con regex ni K4 pueden decir nada de esos pasos.
- **Pero la señal es débil y solo existe en los dominios históricos.**
  - En `history` y `hsm` la AUC está en 0.55–0.65. En `politics` es casi azar (0.48–0.57), aunque el 60 % de sus consultas tiene fecha.
  - Dentro del mismo pool, el score del híbrido separa con AUC 0.77–0.89.
- **En los dominios históricos la fusión sí sube algo** (+0.007 a +0.011), y en `history` llega al criterio de +0.01 dentro del dominio. Pero hay tres razones para no adoptarla:
  - la mejor combinación se elige entre 9 en el propio dominio, así que la cifra es optimista;
  - la combinación ganadora cambia entre 1a y 1b y, aplicada en el otro subtrack, queda entre −0.003 y +0.005;
  - aunque se aplicara solo en `history` y `hsm`, en la macro de los 13 dominios serían unos +0.001–0.002, lejos del criterio de aceptación (Δ ≥ +0.01).
- **Por qué ayuda tan poco.**
  - el híbrido ya trae documentos del mismo tema, y por tanto del mismo periodo;
  - dentro del top-100 el periodo apenas distingue: la similitud media de los no-gold ya es alta (0.64–0.66 en `history`);
  - el paper muestra que el *focus time* sacado de palabras tiene ~20 años de error, así que fecha "la época" y no la década, justo lo que el híbrido ya comparte.
- **Gold con menos o más fechas no es la clave.** Que un gold "hable del periodo" no lo distingue de sus vecinos del pool, que hablan del mismo periodo. Coincide con el EDA 2.5–2.6 (la densidad temporal no separa) y con O5.
- **Las palabras marcadoras no sirven como señal.** Lo más "fechable" del corpus es bibliografía y metadatos, no lenguaje de razonamiento temporal.

## 4. Lo que no se probó, y por qué

| Variante | Qué haría | Por qué no se hace |
|---|---|---|
| `A_con` del paper (o PPR sobre el grafo n-grama ↔ bin) | dar más pasos en el grafo, para que las palabras raras hereden el tiempo de sus vecinas | da **cobertura**, y la cobertura ya es ~100 % (perfil I). Lo que falta es **separación** |
| Pesos TextRank (`S_TR`) | pesar las palabras del documento por su centralidad (PageRank en el grafo del documento) | cambia qué palabras forman el perfil del documento, no el hecho de que los candidatos del pool comparten periodo |
| Aprender el léxico de los gold | quedarse con las palabras temporales que distinguen gold de no-gold | O5 ya lo hizo (log-odds gold frente a hard negatives) y solo salieron términos temáticos y bibliografía; en test no hay gold |
| Expansión de consulta con palabras del periodo | traer documentos de fuera del top-100 ("1830s" → "Jacksonian", "Whig") | es K4 con más ruido. K4, que es la versión limpia, ya no ayuda ([`RESULTADOS_EXPERIMENTOS.md`](RESULTADOS_EXPERIMENTOS.md) §2.9) |

## 5. Error del extractor encontrado de paso

El léxico de `politics` tenía "twenty first" anclado en el siglo I. La causa: en `reteco.temporal`, el ordinal compuesto del patrón `century_word` solo admitía el guion ("twenty-first century"). Sin guion, "twenty first century" casaba solo en "first century".

- **Corrección:** el patrón admite espacio o guion entre las dos palabras, y la clave se normaliza a "twenty-first". "first century AD" sigue siendo el siglo I.
- **Prueba nueva:** `test_compound_ordinal_century` en [`tests/test_extraction.py`](tests/test_extraction.py).
- **Efecto:** despreciable en TT0 (las cifras de §2 ya son con el extractor corregido; con el anterior cambiaban como mucho en el tercer decimal).

Queda sin corregir una limitación conocida: **los siglos sin era** ("the fourth century" en un texto sobre Atenas) se leen como d. C. Resolverlo necesita contexto del documento, no una regla local.

## 6. Conclusiones

1. **El perfil temporal implícito existe y tiene algo de señal**, solo en los dominios históricos (AUC ≈ 0.6) y también sin fechas explícitas.
2. **Sobre el híbrido BM25 + denso aporta poco y de forma inestable**: +0.007 a +0.011 en `history` y `hsm` con la mejor combinación elegida en muestra, que no se traslada entre subtracks; en `politics` empeora. En la macro quedaría muy por debajo del criterio. **No se adopta** ni se hacen las variantes de grafo del paper.
3. **Queda como posible rasgo** para un modelo que ordene los candidatos combinando varias señales, y como la idea temporal más prometedora del proyecto para los dominios históricos: si se retoma, la fusión debería elegirse con validación cruzada y medirse con test pareado. Es barato: 0.5–3 min por dominio para el léxico y ~1 min para los perfiles de un pool.

## 7. Reproducción

```bash
# En el servidor (14 procesos; ~6 min los tres dominios):
.venv/bin/python scripts/temporal_profile_diag.py --domains politics history hsm
```

Salidas en `outputs/exp/perfil_temporal/`: `<dominio>_lex.pkl` (léxico; se reutiliza si existe) y `<dominio>_<subtrack>_stats.pkl` (tablas de §2.2). El script imprime también el léxico (top por especificidad y por fechabilidad) y ejemplos de consultas sin fecha con su perfil.
