"""Expresiones temporales: extracción con expresiones regulares, intervalos y formas gruesas.

Cada expresión ("12 March 1834", "the 1830s", "early 15th century", "300 BC",
"1914–1918") se representa como un intervalo de años ``[start, end)`` con su
granularidad (``TimeSpan``). Los años antes de Cristo son negativos (300 BCE → -300) y
los meses son fracciones de año. Las granularidades van de milenio a día, porque en
TEMPO las consultas hablan tanto de siglos como de fechas concretas.

Partes del módulo:

1. ``TimeSpan``: tipo de dato (intervalo con granularidad).
2. Constructores de intervalos (``year_span``, ``century_span``...).
3. Extracción con regex (``extract_spans``) y papel de cada fecha (``time_role``): cita
   bibliográfica, metadato de la página o fecha del contenido.
4. Formas gruesas de las fechas de un texto para expandir consultas (``coarse_forms``, K4).

Lo usan K4 (``reteco.queries``, ``scripts/k4_diag.py``) y el perfil temporal implícito
(``scripts/temporal_profile_diag.py``). Solo usa la biblioteca estándar.
"""

import math
import re
from dataclasses import dataclass

# Granularidades posibles de un TimeSpan, de más gruesa a más fina, más dos especiales:
# "range" (intervalo arbitrario, "1914-1918") y "present" (la actualidad, "currently").
GRANS = ("millennium", "century", "decade", "year", "quarter", "month", "day", "range", "present")
PRESENT_YEAR = 2025  # "hoy" de los corpus de StackExchange del dataset


@dataclass(frozen=True)
class TimeSpan:
    """Intervalo ``[start, end)`` en años (fraccionarios) con su granularidad.

    Ejemplos: 1834 → ``TimeSpan(1834, 1835, "year")``; marzo de 1834 →
    ``TimeSpan(1834.1667, 1834.25, "month")``; siglo XIX → ``TimeSpan(1800, 1900, "century")``.
    Es inmutable y hashable, así que se puede usar como clave o en conjuntos.

    Attributes:
        start: inicio (incluido), en años; negativo antes de Cristo.
        end: fin (excluido).
        gran: granularidad, una de ``GRANS``.
    """

    start: float
    end: float
    gran: str

    @property
    def length(self) -> float:
        """Duración del intervalo, en años."""
        return self.end - self.start


# ----------------------------------------------------------------------------- constructores


def year_span(y: int) -> TimeSpan:
    """Año ``y`` completo: ``[y, y + 1)``."""
    return TimeSpan(y, y + 1, "year")


def month_span(y: int, m: int) -> TimeSpan:
    """Mes ``m`` (1–12) del año ``y``, como doceavo de año."""
    return TimeSpan(y + (m - 1) / 12, y + m / 12, "month")


def day_span(y: int, m: int, d: int) -> TimeSpan:
    """Día ``d`` del mes ``m`` del año ``y``.

    Aproximación: cada mes es 1/12 de año y cada día 1/365. Basta para decidir
    pertenencias a años, meses o décadas; no es un calendario exacto.
    """
    start = y + (m - 1) / 12 + (d - 1) / 365
    return TimeSpan(start, start + 1 / 365, "day")


def quarter_span(y: int, q: int) -> TimeSpan:
    """Trimestre ``q`` (1–4) del año ``y``."""
    return TimeSpan(y + (q - 1) / 4, y + q / 4, "quarter")


def decade_span(y: int) -> TimeSpan:
    """Década que empieza en ``y`` (``y`` ya redondeado a decena): ``[y, y + 10)``."""
    return TimeSpan(y, y + 10, "decade")


def century_span(n: int, bce: bool = False) -> TimeSpan:
    """Siglo n-ésimo con la convención "1800s": el siglo XIX es [1800, 1900)."""
    if bce:
        return TimeSpan(-n * 100, -(n - 1) * 100, "century")
    return TimeSpan((n - 1) * 100, n * 100, "century")


def range_span(a: float, b: float) -> TimeSpan:
    """Rango de años ``a``–``b`` con los dos extremos incluidos (en cualquier orden)."""
    return TimeSpan(min(a, b), max(a, b) + 1, "range")


def present_span(window: int = 10) -> TimeSpan:
    """La "actualidad": los últimos ``window`` años hasta ``PRESENT_YEAR`` incluido."""
    return TimeSpan(PRESENT_YEAR - window, PRESENT_YEAR + 1, "present")


def _part(span: TimeSpan, part: str | None) -> TimeSpan:
    """Tercio del intervalo para "early / mid / late" (early 19th century → [1800, 1833))."""
    if not part:
        return span
    third = span.length / 3
    k = {"early": 0, "mid": 1, "middle": 1, "late": 2}[part.lower()]
    return TimeSpan(span.start + k * third, span.start + (k + 1) * third, span.gran)


# ----------------------------------------------------------------------------- extracción con regex

# Nombres de mes → número, con las abreviaturas de 3 letras y "sept".
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9
# Los nombres más largos primero, para que "september" no case como "sep".
_MONTH_RE = r"(?P<mon>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
# Ordinales en palabra para "nineteenth century". El compuesto se guarda con guion
# ("twenty-first"); en el patrón casa también con espacio ("twenty first"), ver _ORDINAL_RE.
_ORDINAL_WORDS = {w: i + 1 for i, w in enumerate(
    ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
     "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
     "eighteenth", "nineteenth", "twentieth", "twenty-first"])}
# Alternativa de ordinales para el patrón "century_word", de la más larga a la más corta.
# El guion del compuesto admite también espacios: sin esto, "twenty first century" casaba
# solo en "first century" y salía el siglo I en lugar del XXI.
_ORDINAL_RE = "|".join(w.replace("-", r"[\s-]+") for w in sorted(_ORDINAL_WORDS, key=len, reverse=True))
# Marcas de era: antes de Cristo y después de Cristo.
_ERA_BCE = r"(?:BCE|BC|B\.C\.E?\.?)"
_ERA_CE = r"(?:CE|AD|A\.D\.|C\.E\.)"
# Prefijo opcional "early / mid / late" (se convierte en un tercio del intervalo, ver _part).
_PART = r"(?:(?P<part>early|mid|middle|late)[\s-]+)?"
# Año suelto plausible: 1000–2039. Fuera de ahí un número de 4 cifras casi nunca es un año en el corpus.
_YEAR = r"(?:1[0-9]{3}|20[0-3][0-9])"
# Números que no son años: cantidades con unidad o símbolo, o subapartados pegados al
# número ("§ 1464(t)"). Salen de una revisión manual de una muestra del corpus.
_NOT_YEAR_AFTER = re.compile(
    r"\s*(?:%|percent|people|persons|men|soldiers|troops|km|kilometers|miles|meters|feet|ft|kg|tons|"
    r"usd|dollars|euros|pounds|btc|xmr|ada|eth|bitcoins?|coins|tokens|sats|satoshis|blocks?|bytes|mb|gb|tb|"
    r"words|pages|votes|members|seats|employees|users|students|patients|deaths|copies|units|shares|acres|"
    r"hours|minutes|lbs|transactions|nodes|lines|rpm|mhz|ghz|x\b)"
    r"|\((?:[a-z]|[ivx]{1,4}|\d{1,2})\)", re.IGNORECASE)
# Números que no son años por la palabra que los precede: página, sección, formulario,
# identificador ("printed page 1654", "Form 1099", "uid=1738"). Se mira en los ~25
# caracteres previos. "post" no está porque "post-1945" es una marca temporal.
_NOT_YEAR_BEFORE = re.compile(
    r"(?:\b(?:page|pages|volume|section|article|chapter|form|item|issue|id|uid|number|figure|table|box|"
    r"suite|room|flight|route|model|version|ticket|patent|isbn|issn|doi)|"
    r"\b(?:p|pp|no|nos|vol|vols|sec|art|ch|chap|nr|fig)\.|"
    r"\b[23]d(?:\s+\d+,)?|\bsupp\.|"                     # repertorios judiciales: "F.2d 1284, 1307-08"
    r"§|#|=|[$£€])\s*(?:no\.?\s*)?$", re.IGNORECASE)
# URLs, correos y etiquetas BBCode de spam ("[url=...]"): dentro de ellos no se buscan
# fechas ("renault-trafic-dci-2004/", "uid=1738").
URL_RE = re.compile(
    r"\[/?(?:url|b|i|img|link)[^\]]*\]|https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+|"
    r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|io|info|biz|co\.uk|de|fr|ru)(?:/\S*)?", re.IGNORECASE)

# Patrones de expresiones temporales, de más específico a más general: extract_spans los
# aplica en este orden y un tramo ya reconocido no se reutiliza, así que "12 March 1834"
# lo recoge "dmy" y "year" ya no puede quedarse con "1834".
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("iso", re.compile(r"\b(?P<y>\d{4})-(?P<m>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12]\d|3[01])\b")),
    ("dmy", re.compile(r"\b(?P<d>[0-3]?\d)(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_RE + r",?\s+(?P<y>\d{3,4})\b", re.I)),
    ("mdy", re.compile(r"\b" + _MONTH_RE + r"\s+(?P<d>[0-3]?\d)(?:st|nd|rd|th)?,?\s+(?P<y>\d{3,4})\b", re.I)),
    ("my", re.compile(r"\b" + _MONTH_RE + r",?\s+(?:of\s+)?(?P<y>" + _YEAR + r")\b", re.I)),
    ("quarter", re.compile(r"\b(?:Q(?P<q>[1-4])\s*(?:of\s+)?(?:FY\s*)?(?P<y>" + _YEAR + r")|(?P<y2>" + _YEAR
                           + r")\s*-?\s*Q(?P<q2>[1-4]))\b", re.I)),
    ("century", re.compile(r"\b" + _PART + r"(?P<n>\d{1,2})(?:st|nd|rd|th)[\s-]+centur(?:y|ies)\b(?:\s*(?P<era>"
                           + _ERA_BCE + "|" + _ERA_CE + r"))?", re.I)),
    ("century_word", re.compile(r"\b" + _PART + r"(?P<w>" + _ORDINAL_RE
                                + r")[\s-]+centur(?:y|ies)\b(?:\s*(?P<era>" + _ERA_BCE + "|" + _ERA_CE + r"))?", re.I)),
    ("hundreds", re.compile(r"\b" + _PART + r"(?:the\s+)?(?P<c>1[0-9])00s\b", re.I)),
    ("decade", re.compile(r"\b" + _PART + r"(?:the\s+)?(?P<y>1[0-9]{2}0|20[0-2]0)s\b", re.I)),
    ("decade_short", re.compile(r"(?<![\w'’])['’](?P<d>[0-9])0s\b")),
    ("between", re.compile(r"\bbetween\s+(?P<a>\d{3,4})\s*(?P<ea>" + _ERA_BCE + r")?\s+and\s+(?P<b>\d{3,4})\s*(?P<eb>"
                           + _ERA_BCE + r")?\b", re.I)),
    # Sin "$" en el lookbehind: "$1200-1500" tiene que casar aquí para que _NOT_YEAR_BEFORE
    # lo descarte y consuma el tramo; si no, el patrón "year" se queda con "1500".
    ("range", re.compile(r"(?<![\d#.,/:=@])\b(?:from\s+)?(?P<a>" + _YEAR
                         + r")\s*(?:–|—|-|to|until|till|through)\s*(?P<b>\d{2,4})\b(?![-–/]\d)", re.I)),
    ("era", re.compile(r"\b(?P<y>\d{1,4})\s*(?P<era>" + _ERA_BCE + "|" + _ERA_CE + r")(?![A-Za-z])|\b(?P<era2>AD|A\.D\.)\s*(?P<y2>\d{1,4})\b")),
    ("year", re.compile(r"(?<![\d$£€#.,/:=@])\b(?P<y>" + _YEAR + r")\b(?![.,/:-]?\d)")),
]
# Tipos de patrón que pueden ser un número cualquiera (y no una fecha) según el contexto.
_BARE_NUMBER_KINDS = ("range", "year")
_NOT_A_DATE = TimeSpan(0, 0, "not-a-date")  # centinela: el tramo se consume pero no se devuelve


def _make_span(kind: str, m: re.Match) -> TimeSpan | None:
    """Convierte una coincidencia de ``_PATTERNS`` en ``TimeSpan``.

    Args:
        kind: nombre del patrón que casó (``"iso"``, ``"dmy"``, ``"century"``, ``"year"``...).
        m: la coincidencia.

    Returns:
        - el ``TimeSpan`` de la expresión;
        - ``_NOT_A_DATE`` si es un número que no es una fecha (página, importe…): el
          tramo se marca como usado pero no se devuelve;
        - ``None`` si no es válida (día 45, siglo 30…): el tramo queda libre para otros patrones.
    """
    g = m.groupdict()
    part = g.get("part")
    if kind == "iso":
        return day_span(int(g["y"]), int(g["m"]), int(g["d"]))
    if kind in ("dmy", "mdy"):
        d = int(g["d"])
        if not 1 <= d <= 31:
            return None
        return day_span(int(g["y"]), _MONTHS[g["mon"].lower()], d)
    if kind == "my":
        return month_span(int(g["y"]), _MONTHS[g["mon"].lower()])
    if kind == "quarter":
        y, q = (g["y"], g["q"]) if g["y"] else (g["y2"], g["q2"])
        return quarter_span(int(y), int(q))
    if kind in ("century", "century_word"):
        # "twenty first" / "Twenty - first" → clave canónica "twenty-first".
        n = int(g["n"]) if kind == "century" else _ORDINAL_WORDS[re.sub(r"[\s-]+", "-", g["w"].lower())]
        if not 1 <= n <= 21:
            return None
        bce = bool(g.get("era")) and re.fullmatch(_ERA_BCE, g["era"], re.I) is not None
        return _part(century_span(n, bce), part)
    if kind == "hundreds":
        return _part(TimeSpan(int(g["c"]) * 100, int(g["c"]) * 100 + 100, "century"), part)
    if kind == "decade":
        return _part(decade_span(int(g["y"]) // 10 * 10), part)
    if kind == "decade_short":
        return decade_span(1900 + int(g["d"]) * 10)
    if kind == "between":
        a, b = int(g["a"]), int(g["b"])
        a, b = (-a if g["ea"] else a), (-b if g["eb"] else b)   # cada extremo puede llevar BCE
        return range_span(a, b) if a != b and abs(b - a) <= 3000 else None
    if kind in _BARE_NUMBER_KINDS and _NOT_YEAR_BEFORE.search(m.string, max(0, m.start() - 25), m.start()):
        return _NOT_A_DATE
    if kind == "range":
        if _NOT_YEAR_AFTER.match(m.string, m.end()):
            return _NOT_A_DATE
        a, b_raw = int(g["a"]), g["b"]
        # Segundo extremo abreviado: "1914-18" → 1918 (se completa con las primeras cifras de a).
        b = int(b_raw) if len(b_raw) >= 3 else int(str(a)[:4 - len(b_raw)] + b_raw)
        # Un rango creciente y de como mucho 300 años; si no, es otra cosa ("1200-1100", códigos).
        return range_span(a, b) if a < b <= a + 300 else None
    if kind == "era":
        y = int(g["y"] or g["y2"])
        era = g["era"] or g["era2"]
        if y == 0:   # no existe el año 0
            return None
        return year_span(-y if re.fullmatch(_ERA_BCE, era, re.I) else y)
    if kind == "year":
        if _NOT_YEAR_AFTER.match(m.string, m.end()):
            return _NOT_A_DATE
        return year_span(int(g["y"]))
    return None


# ----------------------------------------------------------------------------- papel de la fecha
#
# Una fecha bien reconocida no siempre es el momento de un hecho del documento:
# - ``citation``: año de una referencia bibliográfica o judicial ("(Oxford, 1964)",
#   "Wang et al., 2016", "Poterba, J.M., 1992", "[2014] UNSWLawJl 39").
# - ``metadata``: fecha de la página y no del contenido ("accessed 2 May 2013",
#   comentarios de foro "August 4, 2025 at 7:09 pm", "© 2004").
# No se descartan por defecto: ``extract_spans(..., drop=...)`` permite quitarlas cuando
# solo interesan las fechas del contenido.

ROLES = ("citation", "metadata")
# Palabras que preceden a una fecha de la página: "accessed", "last updated", "Posted on", "©".
_META_BEFORE = re.compile(
    r"(?:\b(?:accessed|retrieved|consulted|visited|viewed)(?:\s+on)?|"
    r"\blast\s+(?:updated|modified|edited|revised|accessed)(?:\s+on)?|"
    r"\b(?:updated|modified|posted|submitted|commented|replied|answered|asked)\s+on|"
    r"\b(?:date|updated|modified|posted|published)\s*:|"
    r"©|\(c\)|\bcopyright(?:\s+©)?)\s*:?\s*$", re.IGNORECASE)
_META_AFTER = re.compile(r"\s*(?:,|at|@|-)?\s*\d{1,2}:\d{2}\b", re.IGNORECASE)    # hora del comentario
# "Wang et al., 2016" / "Smith et al. (2018".
_ETAL_BEFORE = re.compile(r"\bet\s+al\.?,?\s*\(?\s*$", re.IGNORECASE)
# "Poterba, J.M." — cada inicial va seguida de espacio, coma, fin u otra inicial, para que
# "England, D.Sc., 1949" (un título académico) no cuente como referencia.
_REFLIST_BEFORE = re.compile(r"\b[A-Z][\w'’-]+,\s+(?:[A-Z]\.(?:\s|,|$|(?=[A-Z]\.))){1,3}")
# Años de titulación ("(MBBCh 1965)", "Ph.D., 1942", "(BArch 1955)"): son hechos biográficos.
_DEGREE_BEFORE = re.compile(
    r"\b(?:B\.?A|M\.?A|B\.?Sc|M\.?Sc|D\.?Sc|Ph\.?\s?D|M\.?D|J\.?D|LL\.?[BM]|MBA|MBBCh|MBBS|BArch|MArch|"
    r"B\.?Ed|M\.?Ed|B\.?Eng|M\.?Eng|D\.?Phil|M\.?Phil)\.?,?\s*$")
# Apellido: palabra que empieza en mayúscula (admite apóstrofos y guiones: "O'Brien", "Lévi-Strauss").
_SURNAME = r"[A-Z][\w'’-]+"
_AUTHOR_IN_PAREN = re.compile(                                # "(see Feferman", "(Head", "(Oxford,", "(Blinder, Goodhart, & Wyplosz,"
    r"^\s*(?:see\s+(?:also\s+)?|cf\.\s+|e\.g\.,?\s+)?" + _SURNAME
    + r"(?:,\s*" + _SURNAME + r")*(?:,?\s+(?:and|&)\s+" + _SURNAME + r"|\s+et\s+al\.?)?,?\s*$")
_PUBLISHER_WORDS = r"(?:press|university|publishing|publishers?|pub|books|verlag|society|commission|office|co|inc|ltd)"
_PUBLISHER_IN_PAREN = re.compile(                             # "(London: Z ed,", "(… University Press,", "(Prince Society, Boston,"
    r"(?::\s*[^:()\[\]]{1,60},\s*$)|\b(?:ed|eds)\.?,\s*$|\b" + _PUBLISHER_WORDS + r"\b\.?,(?:[^,()]{1,30},)?\s*$",
    re.IGNORECASE)
# "Toronto: University of Toronto Press, 2011." / "Den Haag: Martinus Nijhoff, 1952;" sin paréntesis.
# La "ciudad" no puede ser una etiqueta ("Note:", "Source:"…) para no confundir notas con pies de imprenta.
_IMPRINT_BEFORE = re.compile(
    r"(?:^|[.,;(\[]\s*)(?!(?:Note|Notes|Source|Sources|See|Date|Image|Photo|Figure|Table|Example|Caption)\b)"
    r"[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+){0,2}:\s+[A-Z][^:;()\[\]]{2,60},\s*$")
# "Peacock, Darren (2002)" / "Li, David (2000)": entrada de una lista de referencias.
_REFENTRY_BEFORE_PAREN = re.compile(_SURNAME + r",\s+" + _SURNAME + r"$")
# Tras el año: volumen o página ("1802, Vol.", "(2015) p."), editor ("2016 edited by") o
# referencia Vancouver ("2010; 17:1318-1325").
_REF_AFTER = re.compile(
    r"[a-z]?\)?\s*(?:,\s*(?:vols?\.|issue\b)|,?\s*pp?\.\s*\d|\s+(?:ed\.|eds\.|edited\s+by)|;\s*\d+(?:\(\d+\))?:\s*\d+)",
    re.IGNORECASE)
# Tribunal antes del año dentro del paréntesis: "(1st Dept 2003)", "(9th Cir. 1999)".
_COURT_IN_PAREN = re.compile(r"\b(?:\d+(?:st|nd|rd|th)\s+(?:dept|cir)\.?|cir\.|app\.|ct\.)\s*$", re.IGNORECASE)
# El año cierra la referencia: ")" / "]", o ";" si sigue otra referencia en el mismo paréntesis.
_CLOSE_AFTER = re.compile(r"[a-z]?(?:\s*[,:]\s*(?:pp?\.\s*)?\d+(?:\s*[-–]\s*\d+)?)?\s*[)\];]")
# Señales que no necesitan paréntesis:
_PAGE_AFTER = re.compile(r"[a-z]?\s*,\s*(?:pp?\.|pages?\b)", re.IGNORECASE)            # "Long 2005, p. 3"
_EDITION_BEFORE = re.compile(r"(?:\b\d+(?:st|nd|rd|th)\s+ed(?:ition)?\.?|\b\d+\s+vols?\.?),\s*$", re.IGNORECASE)
# Sin regla para listas "Simmel, 1900; Crump, 1978": tienen la misma forma que las carreras de
# genealogy ("Lieutenant, 1904; Commander, 1916;"), que son hechos biográficos.
# Apellido (con "et al." opcional) justo antes del año, y cierre de paréntesis justo
# después: se usan juntos para los paréntesis abiertos en la oración anterior.
_AUTHOR_BEFORE = re.compile(r"\b[A-Z][\w'’-]+(?:\s+et\s+al\.?)?,?\s*$")
_ORPHAN_CLOSE = re.compile(r"[a-z]?\s*[)\]]")
_PAGES_BEFORE_PAREN = re.compile(r"(?:\d+\s*[-–]\s*\d+|,\s*e?\d+),?$")                 # "505–512 (1995)", "8, e61912 (2013)"


def time_role(text: str, s: int, e: int, kind: str = "year") -> str | None:
    """Papel de la expresión ``text[s:e]``: ``"citation"``, ``"metadata"`` o ``None`` (fecha del contenido).

    ``metadata`` se comprueba en todas las expresiones; ``citation``, solo en años sueltos
    (las fechas completas de las referencias son raras y ambiguas).

    Args:
        text: texto completo (normalmente una oración).
        s: inicio de la expresión en ``text``.
        e: fin de la expresión.
        kind: patrón de ``_PATTERNS`` que la reconoció.

    Returns:
        ``"citation"``, ``"metadata"`` o ``None``. Las reglas están en el orden en que se
        comprueban; la primera que se cumple decide.
    """
    before = text[max(0, s - 100):s]   # contexto previo: hasta 100 caracteres
    if _META_BEFORE.search(before[-40:]) or _META_AFTER.match(text, e):
        return "metadata"
    # Las citas solo se buscan en años sueltos, y nunca tras un título académico ("PhD, 1942").
    if kind != "year" or _DEGREE_BEFORE.search(before):
        return None
    # Señales de referencia que no necesitan paréntesis.
    if (_ETAL_BEFORE.search(before) or _REFLIST_BEFORE.search(before[-60:])
            or _PAGE_AFTER.match(text, e) or _EDITION_BEFORE.search(before)
            or _IMPRINT_BEFORE.search(before) or _REF_AFTER.match(text, e)):
        return "citation"
    # ¿Está el año dentro de un paréntesis o corchete que se cierra justo después?
    op = max(before.rfind("("), before.rfind("["))
    if op < 0 or max(before.rfind(")"), before.rfind("]")) > op:
        # Paréntesis que se abrió en la "oración" anterior (el divisor corta en "pp. 33"):
        # "Boutroux 1908)." cuenta como referencia si el año sigue a un apellido.
        if op < 0 and _ORPHAN_CLOSE.match(text, e) and _AUTHOR_BEFORE.search(before):
            return "citation"
        return None
    # Con varias referencias en el mismo paréntesis ("(Head 2019; Mandiant 2016)") cuenta la última.
    inside, close = before[op + 1:].split(";")[-1], _CLOSE_AFTER.match(text, e)
    if close is None:
        return None
    if _AUTHOR_IN_PAREN.search(inside) or _PUBLISHER_IN_PAREN.search(inside) or _COURT_IN_PAREN.search(inside):
        return "citation"
    # Paréntesis con solo el año, "(1998)": decide lo que hay antes y después del paréntesis.
    if inside.strip() == "":
        prev = before[:op].rstrip()
        after_close = text[close.end():close.end() + 30]
        if (re.search(r"(?:['’]s|\bet\s+al\.?)$", prev)             # "Lee’s (1998)", "Smith et al. (2018)"
                or prev == "" or prev[-1] in ".;"                  # "(2017)." como fragmento de referencia
                or _PAGES_BEFORE_PAREN.search(prev)                 # "25(3), 505–512 (1995)"
                or _REFENTRY_BEFORE_PAREN.search(prev)              # "Peacock, Darren (2002)"
                or re.match(r"\s*\d+\s*\(", after_close)            # "(2014) 37(3) UNSW Law Journal"
                or (before[op] == "[" and re.match(r"\s*[A-Z][A-Za-z]*[A-Z]", after_close))):  # "[2014] UNSWLawJl"
            return "citation"
    return None


def extract_spans_with_role(text: str, roles: bool = True) -> list[tuple[int, int, TimeSpan, str | None]]:
    """Como ``extract_spans``, pero devuelve también el papel de cada expresión (``time_role``).

    Con ``roles=False`` no se calcula el papel (siempre ``None``): es más rápido y es lo
    que usa ``extract_spans`` cuando no hay que descartar nada.

    Args:
        text: texto libre.
        roles: si se calcula el papel de cada expresión.

    Returns:
        ``[(inicio, fin, TimeSpan, papel)]`` ordenado por posición.
    """
    # taken[i] = 1 si el carácter i ya pertenece a una expresión (o a una URL).
    taken = bytearray(len(text))
    for m in URL_RE.finditer(text):                    # en URLs no hay fechas del contenido
        taken[m.start():m.end()] = b"\x01" * (m.end() - m.start())
    out = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(taken[s:e]):   # solapa con algo ya reconocido por un patrón más específico
                continue
            span = _make_span(kind, m)
            if span is None:
                continue
            # Un número descartado (página, importe…) consume su tramo para que un patrón
            # más general no recoja un trozo ("$1200-1500" no deja "1500" como año).
            taken[s:e] = b"\x01" * (e - s)
            if span is _NOT_A_DATE:
                continue
            out.append((s, e, span, time_role(text, s, e, kind) if roles else None))
    return sorted(out, key=lambda x: x[0])


def extract_spans(text: str, drop: tuple[str, ...] | frozenset[str] = ()) -> list[tuple[int, int, TimeSpan]]:
    """Expresiones temporales de un texto: ``[(inicio, fin, TimeSpan)]`` ordenadas por posición.

    Los patrones se aplican de más específico a más general; un tramo ya reconocido
    no se vuelve a usar ("March 1834" no produce además "1834"). No se buscan fechas
    dentro de URLs, y los números con contexto de página, sección, importe o
    identificador no cuentan como años.

    Args:
        text: texto libre.
        drop: papeles que se descartan (subconjunto de ``ROLES``): ``"citation"`` y/o
            ``"metadata"``. Por defecto no se descarta nada.

    Returns:
        ``[(inicio, fin, TimeSpan)]`` ordenado por posición.

    Raises:
        ValueError: si ``drop`` contiene un papel que no está en ``ROLES``.
    """
    unknown = set(drop) - set(ROLES)
    if unknown:
        raise ValueError(f"papeles desconocidos: {sorted(unknown)}; válidos: {ROLES}")
    # Si no hay nada que descartar, no se calcula el papel (time_role es la parte lenta).
    return [(s, e, span) for s, e, span, role in extract_spans_with_role(text, roles=bool(drop)) if role not in drop]


# ----------------------------------------------------------------------------- formas gruesas (K4)
#
# K4 (PLAN_EXPERIMENTOS.md §6.2, RESULTADOS_EXPERIMENTOS.md §2.9): a una consulta con "1834" se le
# añaden "1830s" y "19th century", porque un gold puede hablar del periodo sin repetir el
# año exacto. Solo se añaden términos; nunca se filtra nada.

COARSE_MIN_YEAR = 1000      # por debajo no se usan décadas ("the 830s" no aparece en el corpus)
CENTURY_MAX_YEAR = 2000     # desde 2000 no se añade "21st century": casi todo el corpus es de este siglo
MAX_COARSE_SPAN = 300       # rangos más largos ("1500–1900") son demasiado vagos para expandir


def ordinal(n: int) -> str:
    """Ordinal numérico en inglés: 1 → ``"1st"``, 2 → ``"2nd"``, 11 → ``"11th"``, 21 → ``"21st"``."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _decade_form(y: int) -> str | None:
    """``"1830s"`` para cualquier año de esa década; ``None`` fuera de 1000–2039.

    Tampoco hay década para 1800–1809 y similares: "the 1800s" se lee como el siglo
    (el extractor lo trata así, patrón ``hundreds``), y el siglo ya se añade aparte.
    "2000s" sí es la década.
    """
    if not COARSE_MIN_YEAR <= y < 2040 or (y < 2000 and y % 100 < 10):
        return None
    return f"{y // 10 * 10}s"


def _century_form(y: int) -> str | None:
    """``"19th century"`` o ``"4th century BC"`` con la convención de ``century_span``.

    El siglo XIX es [1800, 1900) y el siglo III a. C. es [-300, -200). Devuelve ``None``
    desde ``CENTURY_MAX_YEAR``.
    """
    if y >= CENTURY_MAX_YEAR:
        return None
    if y >= 0:
        return f"{ordinal(y // 100 + 1)} century"
    return f"{ordinal(math.ceil(-y / 100))} century BC"


def coarse_forms(text: str) -> list[str]:
    """Formas más gruesas de las expresiones temporales de un texto (K4).

    - año, mes, día o trimestre → su década y su siglo ("12 March 1834" → "1830s", "19th century");
    - década → su forma canónica y su siglo ("'60s" → "1960s", "20th century");
    - siglo → su forma numérica ("nineteenth century" o "the 1800s" → "19th century");
    - rango → las décadas que cubre (si dura ≤ 31 años) y sus siglos ("1914–1918" → "1910s", "20th century").

    Se omiten la actualidad ("currently"), los milenios, los rangos de más de
    ``MAX_COARSE_SPAN`` años y las formas que ya aparecen en el texto.

    Args:
        text: texto de la consulta (o de la consulta más el paso, en 1b).

    Returns:
        Formas nuevas, sin repetir y en orden de aparición.
    """
    out: list[str] = []
    for _, _, span in extract_spans(text):
        if span.gran in ("present", "millennium") or span.length > MAX_COARSE_SPAN:
            continue
        first, last = math.floor(span.start), math.ceil(span.end) - 1
        with_decade = span.gran != "century" and (span.gran != "range" or span.length <= 31)
        # Los extremos y un año por década bastan para cubrir todas las décadas y siglos del tramo.
        years = sorted({first, last, *range(first - first % 10, last + 1, 10)} & set(range(first, last + 1)))
        for y in years:
            if with_decade:
                out.append(_decade_form(y))
            out.append(_century_form(y))
    lower = text.lower()
    seen: set[str] = set()
    forms = []
    for f in out:
        if f and f not in seen and f.lower() not in lower:
            seen.add(f)
            forms.append(f)
    return forms

