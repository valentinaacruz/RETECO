"""Expresiones temporales: extracción con reglas, intervalos, jerarquía de tiempo y emparejamiento.

TG-RAG (``papers/2510.13590v1.pdf`` §III-A) organiza los timestamps en una jerarquía
(año → trimestre → mes → día). En TEMPO las granularidades útiles son más gruesas, así
que aquí la jerarquía es milenio → siglo → década → año → trimestre → mes → día.

Cada expresión se representa como un intervalo de años ``[start, end)`` con su
granularidad (``TimeSpan``). Los años antes de Cristo son negativos (300 BCE → -300) y
los meses son fracciones de año. Que τ pertenezca a ``T_q`` se decide con aritmética de
intervalos, que equivale a la pertenencia de nodos en la jerarquía del paper.

Solo usa la biblioteca estándar.
"""

import math
import re
from dataclasses import dataclass

GRANS = ("millennium", "century", "decade", "year", "quarter", "month", "day", "range", "present")
PRESENT_YEAR = 2025  # "hoy" de los corpus de StackExchange del dataset


@dataclass(frozen=True)
class TimeSpan:
    """Intervalo ``[start, end)`` en años (fraccionarios) con su granularidad."""

    start: float
    end: float
    gran: str

    def to_str(self) -> str:
        return f"{self.gran}:{self.start:g}:{self.end:g}"

    @staticmethod
    def from_str(s: str) -> "TimeSpan":
        gran, start, end = s.rsplit(":", 2)
        return TimeSpan(float(start), float(end), gran)

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Scope:
    """Alcance temporal de una consulta: ``logic`` ∈ {at, before, after} sobre ``span``.

    ``between`` se representa como ``at`` sobre un ``TimeSpan`` de tipo ``range``.
    """

    logic: str
    span: TimeSpan

    def interval(self) -> tuple[float, float]:
        if self.logic == "before":
            return -math.inf, self.span.start
        if self.logic == "after":
            return self.span.end, math.inf
        return self.span.start, self.span.end

    def to_str(self) -> str:
        return f"{self.logic}|{self.span.to_str()}"

    @staticmethod
    def from_str(s: str) -> "Scope":
        logic, span = s.split("|", 1)
        return Scope(logic, TimeSpan.from_str(span))


# ----------------------------------------------------------------------------- constructores


def year_span(y: int) -> TimeSpan:
    return TimeSpan(y, y + 1, "year")


def month_span(y: int, m: int) -> TimeSpan:
    return TimeSpan(y + (m - 1) / 12, y + m / 12, "month")


def day_span(y: int, m: int, d: int) -> TimeSpan:
    start = y + (m - 1) / 12 + (d - 1) / 365
    return TimeSpan(start, start + 1 / 365, "day")


def quarter_span(y: int, q: int) -> TimeSpan:
    return TimeSpan(y + (q - 1) / 4, y + q / 4, "quarter")


def decade_span(y: int) -> TimeSpan:
    return TimeSpan(y, y + 10, "decade")


def century_span(n: int, bce: bool = False) -> TimeSpan:
    """Siglo n-ésimo con la convención "1800s": el siglo XIX es [1800, 1900)."""
    if bce:
        return TimeSpan(-n * 100, -(n - 1) * 100, "century")
    return TimeSpan((n - 1) * 100, n * 100, "century")


def range_span(a: float, b: float) -> TimeSpan:
    return TimeSpan(min(a, b), max(a, b) + 1, "range")


def present_span(window: int = 10) -> TimeSpan:
    return TimeSpan(PRESENT_YEAR - window, PRESENT_YEAR + 1, "present")


def _part(span: TimeSpan, part: str | None) -> TimeSpan:
    """Tercio del intervalo para "early / mid / late" (early 19th century → [1800, 1833))."""
    if not part:
        return span
    third = span.length / 3
    k = {"early": 0, "mid": 1, "middle": 1, "late": 2}[part.lower()]
    return TimeSpan(span.start + k * third, span.start + (k + 1) * third, span.gran)


# ----------------------------------------------------------------------------- jerarquía


def node_key(gran: str, start: float) -> str:
    """Clave del nodo de tiempo de granularidad ``gran`` que contiene el instante ``start``."""
    y = math.floor(start)
    if gran == "millennium":
        return f"millennium:{y // 1000 * 1000}"
    if gran == "century":
        return f"century:{y // 100 * 100}"
    if gran == "decade":
        return f"decade:{y // 10 * 10}"
    if gran == "year":
        return f"year:{y}"
    frac = start - y
    if gran == "quarter":
        return f"quarter:{y}-Q{min(int(frac * 4) + 1, 4)}"
    month = min(int(frac * 12 + 1e-9) + 1, 12)
    if gran == "month":
        return f"month:{y}-{month:02d}"
    day = int(round((frac - (month - 1) / 12) * 365)) + 1
    return f"day:{y}-{month:02d}-{day:02d}"


_HIERARCHY = ("millennium", "century", "decade", "year", "month", "day")


def span_nodes(span: TimeSpan, max_nodes: int = 30) -> list[str]:
    """Nodos de la jerarquía que cubren el intervalo (enlaces entre niveles del grafo).

    Un ``TimeSpan`` de granularidad fija cae en un nodo. Un rango o un "presente" se
    reparte en años si caben en ``max_nodes``; si no, en décadas, siglos o milenios.
    """
    if span.gran in _HIERARCHY or span.gran == "quarter":
        return [node_key(span.gran, span.start)]
    for gran, step in (("year", 1), ("decade", 10), ("century", 100), ("millennium", 1000)):
        first = math.floor(span.start / step) * step
        n = math.ceil((span.end - first) / step)
        if n <= max_nodes:
            return [node_key(gran, first + i * step) for i in range(n)]
    return [node_key("millennium", span.start)]


def ancestors(key: str) -> list[str]:
    """Nodos ancestros de un nodo de tiempo, del más fino al más grueso."""
    gran, label = key.split(":", 1)
    m = re.match(r"(-?\d+)", label)
    if not m:
        return []
    y = int(m.group(1))
    order = ("day", "month", "quarter", "year", "decade", "century", "millennium")
    if gran not in order:
        return []
    out = []
    if gran == "day":
        out.append(f"month:{label[:len(label) - 3]}")
    if gran in ("day", "month"):
        month = int(label.split("-")[-1] if gran == "month" else label.split("-")[-2])
        out.append(f"quarter:{y}-Q{(month - 1) // 3 + 1}")
    for g in order[order.index(gran) + 1:]:
        if g in ("month", "quarter"):
            continue
        out.append(node_key(g, y))
    return out


# ----------------------------------------------------------------------------- extracción con regex

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_MONTH_RE = r"(?P<mon>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_ORDINAL_WORDS = {w: i + 1 for i, w in enumerate(
    ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
     "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
     "eighteenth", "nineteenth", "twentieth", "twenty-first"])}
_ERA_BCE = r"(?:BCE|BC|B\.C\.E?\.?)"
_ERA_CE = r"(?:CE|AD|A\.D\.|C\.E\.)"
_PART = r"(?:(?P<part>early|mid|middle|late)[\s-]+)?"
_YEAR = r"(?:1[0-9]{3}|20[0-3][0-9])"
# Números que no son años: cantidades con unidad o símbolo, o subapartados pegados al
# número ("§ 1464(t)"). Ver REVISION_EXTRACCION_TGRAG.md §3.2.
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
# fechas ("renault-trafic-dci-2004/", "uid=1738"). También lo usa ``reteco.tgraph``.
URL_RE = re.compile(
    r"\[/?(?:url|b|i|img|link)[^\]]*\]|https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+|"
    r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|io|info|biz|co\.uk|de|fr|ru)(?:/\S*)?", re.IGNORECASE)

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("iso", re.compile(r"\b(?P<y>\d{4})-(?P<m>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12]\d|3[01])\b")),
    ("dmy", re.compile(r"\b(?P<d>[0-3]?\d)(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH_RE + r",?\s+(?P<y>\d{3,4})\b", re.I)),
    ("mdy", re.compile(r"\b" + _MONTH_RE + r"\s+(?P<d>[0-3]?\d)(?:st|nd|rd|th)?,?\s+(?P<y>\d{3,4})\b", re.I)),
    ("my", re.compile(r"\b" + _MONTH_RE + r",?\s+(?:of\s+)?(?P<y>" + _YEAR + r")\b", re.I)),
    ("quarter", re.compile(r"\b(?:Q(?P<q>[1-4])\s*(?:of\s+)?(?:FY\s*)?(?P<y>" + _YEAR + r")|(?P<y2>" + _YEAR
                           + r")\s*-?\s*Q(?P<q2>[1-4]))\b", re.I)),
    ("century", re.compile(r"\b" + _PART + r"(?P<n>\d{1,2})(?:st|nd|rd|th)[\s-]+centur(?:y|ies)\b(?:\s*(?P<era>"
                           + _ERA_BCE + "|" + _ERA_CE + r"))?", re.I)),
    ("century_word", re.compile(r"\b" + _PART + r"(?P<w>" + "|".join(sorted(_ORDINAL_WORDS, key=len, reverse=True))
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
        n = int(g["n"]) if kind == "century" else _ORDINAL_WORDS[g["w"].lower()]
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
        a, b = (-a if g["ea"] else a), (-b if g["eb"] else b)
        return range_span(a, b) if a != b and abs(b - a) <= 3000 else None
    if kind in _BARE_NUMBER_KINDS and _NOT_YEAR_BEFORE.search(m.string, max(0, m.start() - 25), m.start()):
        return _NOT_A_DATE
    if kind == "range":
        if _NOT_YEAR_AFTER.match(m.string, m.end()):
            return _NOT_A_DATE
        a, b_raw = int(g["a"]), g["b"]
        b = int(b_raw) if len(b_raw) >= 3 else int(str(a)[:4 - len(b_raw)] + b_raw)
        return range_span(a, b) if a < b <= a + 300 else None
    if kind == "era":
        y = int(g["y"] or g["y2"])
        era = g["era"] or g["era2"]
        if y == 0:
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
# No se descartan por defecto: ``extract_spans(..., drop=...)`` permite medir en TG0 si
# quitarlas ayuda. Ver REVISION_EXTRACCION_TGRAG.md §3.3.

ROLES = ("citation", "metadata")
_META_BEFORE = re.compile(
    r"(?:\b(?:accessed|retrieved|consulted|visited|viewed)(?:\s+on)?|"
    r"\blast\s+(?:updated|modified|edited|revised|accessed)(?:\s+on)?|"
    r"\b(?:updated|modified|posted|submitted|commented|replied|answered|asked)\s+on|"
    r"\b(?:date|updated|modified|posted|published)\s*:|"
    r"©|\(c\)|\bcopyright(?:\s+©)?)\s*:?\s*$", re.IGNORECASE)
_META_AFTER = re.compile(r"\s*(?:,|at|@|-)?\s*\d{1,2}:\d{2}\b", re.IGNORECASE)    # hora del comentario
_ETAL_BEFORE = re.compile(r"\bet\s+al\.?,?\s*\(?\s*$", re.IGNORECASE)
# "Poterba, J.M." — cada inicial va seguida de espacio, coma, fin u otra inicial, para que
# "England, D.Sc., 1949" (un título académico) no cuente como referencia.
_REFLIST_BEFORE = re.compile(r"\b[A-Z][\w'’-]+,\s+(?:[A-Z]\.(?:\s|,|$|(?=[A-Z]\.))){1,3}")
# Años de titulación ("(MBBCh 1965)", "Ph.D., 1942", "(BArch 1955)"): son hechos biográficos.
_DEGREE_BEFORE = re.compile(
    r"\b(?:B\.?A|M\.?A|B\.?Sc|M\.?Sc|D\.?Sc|Ph\.?\s?D|M\.?D|J\.?D|LL\.?[BM]|MBA|MBBCh|MBBS|BArch|MArch|"
    r"B\.?Ed|M\.?Ed|B\.?Eng|M\.?Eng|D\.?Phil|M\.?Phil)\.?,?\s*$")
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
_COURT_IN_PAREN = re.compile(r"\b(?:\d+(?:st|nd|rd|th)\s+(?:dept|cir)\.?|cir\.|app\.|ct\.)\s*$", re.IGNORECASE)
# El año cierra la referencia: ")" / "]", o ";" si sigue otra referencia en el mismo paréntesis.
_CLOSE_AFTER = re.compile(r"[a-z]?(?:\s*[,:]\s*(?:pp?\.\s*)?\d+(?:\s*[-–]\s*\d+)?)?\s*[)\];]")
# Señales que no necesitan paréntesis:
_PAGE_AFTER = re.compile(r"[a-z]?\s*,\s*(?:pp?\.|pages?\b)", re.IGNORECASE)            # "Long 2005, p. 3"
_EDITION_BEFORE = re.compile(r"(?:\b\d+(?:st|nd|rd|th)\s+ed(?:ition)?\.?|\b\d+\s+vols?\.?),\s*$", re.IGNORECASE)
# Sin regla para listas "Simmel, 1900; Crump, 1978": tienen la misma forma que las carreras de
# genealogy ("Lieutenant, 1904; Commander, 1916;"), que son hechos biográficos.
_AUTHOR_BEFORE = re.compile(r"\b[A-Z][\w'’-]+(?:\s+et\s+al\.?)?,?\s*$")
_ORPHAN_CLOSE = re.compile(r"[a-z]?\s*[)\]]")
_PAGES_BEFORE_PAREN = re.compile(r"(?:\d+\s*[-–]\s*\d+|,\s*e?\d+),?$")                 # "505–512 (1995)", "8, e61912 (2013)"


def time_role(text: str, s: int, e: int, kind: str = "year") -> str | None:
    """Papel de la expresión ``text[s:e]``: ``"citation"``, ``"metadata"`` o ``None`` (fecha del contenido).

    ``metadata`` se comprueba en todas las expresiones; ``citation``, solo en años sueltos
    (las fechas completas de las referencias son raras y ambiguas).
    """
    before = text[max(0, s - 100):s]
    if _META_BEFORE.search(before[-40:]) or _META_AFTER.match(text, e):
        return "metadata"
    if kind != "year" or _DEGREE_BEFORE.search(before):
        return None
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
    """
    taken = bytearray(len(text))
    for m in URL_RE.finditer(text):                    # en URLs no hay fechas del contenido
        taken[m.start():m.end()] = b"\x01" * (m.end() - m.start())
    out = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(taken[s:e]):
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
    """
    unknown = set(drop) - set(ROLES)
    if unknown:
        raise ValueError(f"papeles desconocidos: {sorted(unknown)}; válidos: {ROLES}")
    return [(s, e, span) for s, e, span, role in extract_spans_with_role(text, roles=bool(drop)) if role not in drop]


# ----------------------------------------------------------------------------- alcance de la consulta

_BEFORE_RE = re.compile(r"\b(before|prior\s+to|until|till|up\s+to|earlier\s+than|predat\w*|by|pre)\s*[-–]?\s*(the\s+)?$", re.I)
_AFTER_RE = re.compile(r"\b(after|since|post|following|later\s+than|beyond)\s*[-–]?\s*(the\s+)?$", re.I)
_PRESENT_RE = re.compile(r"\b(currently|nowadays|today|these\s+days|at\s+present|presently|as\s+of\s+now|"
                         r"still|anymore|any\s+more|modern[- ]day|recent(ly)?|latest|up[- ]to[- ]date)\b", re.I)


def query_scopes(text: str, present: bool = False) -> list[Scope]:
    """Alcance temporal ``T_q`` de un texto con reglas (equivalente sin LLM al prompt del paper).

    La lógica ``before`` / ``after`` se lee en las 3 palabras previas a cada expresión
    ("prior to 1600", "since the 1830s", "pre-1600"). Los rangos son ``at`` sobre un
    ``range``. Con ``present=True``, las marcas de actualidad ("currently", "still")
    añaden ``at`` sobre los últimos 10 años.
    """
    scopes = []
    for s, _, span in extract_spans(text):
        before = text[max(0, s - 25):s]
        if _BEFORE_RE.search(before):
            logic = "before"
        elif _AFTER_RE.search(before):
            logic = "after"
        else:
            logic = "at"
        scopes.append(Scope(logic, span))
    if present and _PRESENT_RE.search(text):
        scopes.append(Scope("at", present_span()))
    return list(dict.fromkeys(scopes))


_NORM_RE = [
    (re.compile(r"^(-?\d{1,4})-(\d{2})-(\d{2})$"), lambda m: day_span(int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"^(-?\d{1,4})-(\d{2})$"), lambda m: month_span(int(m[1]), int(m[2]))),
    (re.compile(r"^(-?\d{1,4})-?Q([1-4])$", re.I), lambda m: quarter_span(int(m[1]), int(m[2]))),
    (re.compile(r"^(\d{2})00s$"), lambda m: TimeSpan(int(m[1]) * 100, int(m[1]) * 100 + 100, "century")),
    (re.compile(r"^(-?\d{3})0s$"), lambda m: decade_span(int(m[1]) * 10)),
    (re.compile(r"^C(-?\d{1,2})$", re.I), lambda m: century_span(abs(int(m[1])), int(m[1]) < 0)),
    (re.compile(r"^(-?\d{1,4})\s*[/–]\s*(-?\d{1,4})$"), lambda m: range_span(int(m[1]), int(m[2]))),
    (re.compile(r"^(-?\d{1,4})$"), lambda m: year_span(int(m[1]))),
]


def parse_normalized(value: str | None) -> TimeSpan | None:
    """Convierte un timestamp normalizado (salida del LLM) en ``TimeSpan``.

    Formatos: ``1834``, ``-300``, ``1834-03``, ``1834-03-12``, ``2020-Q3``, ``1830s``,
    ``1800s``, ``C19`` (``C-5`` = siglo V a. C.), ``1914/1918``, ``present``. Si no
    encaja, se prueba con ``extract_spans`` sobre el texto libre.
    """
    if not value:
        return None
    v = str(value).strip().strip('"').strip()
    if v.lower() in ("", "null", "none", "unknown", "n/a"):
        return None
    if v.lower() in ("present", "now", "today", "current"):
        return present_span()
    for pat, make in _NORM_RE:
        m = pat.match(v)
        if m:
            try:
                return make(m)
            except (ValueError, KeyError):
                return None
    spans = extract_spans(v)
    return spans[0][2] if spans else None


# ----------------------------------------------------------------------------- emparejamiento τ ↔ T_q


def temporal_match(span: TimeSpan | None, scopes: list[Scope], mode: str = "contained") -> float:
    """Grado en que τ pertenece al alcance ``T_q`` (máximo sobre sus ``Scope``).

    Args:
        span: τ de la arista (``None`` = hecho sin tiempo → 0).
        scopes: alcance de la consulta.
        mode: ``contained`` (1 si τ ⊆ alcance; el indicador del paper), ``overlap``
            (1 si se solapan; acepta ancestros como "19th century" para 1834) o ``soft``
            (fracción de τ dentro del alcance; sin solape decae como
            ``exp(-distancia / escala)``, con escala = max(10 años, longitud de τ)).

    Returns:
        Valor en [0, 1].
    """
    if span is None or not scopes:
        return 0.0
    best = 0.0
    for sc in scopes:
        lo, hi = sc.interval()
        inter = min(span.end, hi) - max(span.start, lo)
        if mode == "contained":
            val = 1.0 if span.start >= lo - 1e-9 and span.end <= hi + 1e-9 else 0.0
        elif mode == "overlap":
            val = 1.0 if inter > 0 else 0.0
        elif mode == "soft":
            if inter > 0:
                val = inter / span.length if span.length > 0 else 1.0
            else:
                gap = max(lo - span.end, span.start - hi)
                val = math.exp(-gap / max(10.0, span.length))
        else:
            raise ValueError(f"modo de emparejamiento desconocido: {mode}")
        best = max(best, val)
        if best >= 1.0:
            break
    return best
