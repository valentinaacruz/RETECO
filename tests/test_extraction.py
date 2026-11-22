"""Pruebas del extractor de expresiones temporales con reglas (``reteco.temporal``).

Los casos salen de una revisión manual de una muestra del corpus: los errores que se
corrigieron y fechas correctas que deben seguir reconociéndose. También prueban las
formas gruesas de K4 (``coarse_forms``). No necesita pytest:

    .venv/bin/python tests/test_extraction.py

(también funciona con ``pytest tests/`` si se instala).
"""

import sys
from pathlib import Path

# Permite importar el paquete reteco (en src/) sin instalarlo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reteco.temporal import coarse_forms, extract_spans, extract_spans_with_role  # noqa: E402


def spans(text: str, drop=()) -> list[tuple[str, str, float]]:
    """``[(texto, granularidad, inicio)]`` de las expresiones encontradas.

    Args:
        text: texto de prueba.
        drop: papeles que se descartan (se pasa a ``extract_spans``).
    """
    return [(text[s:e], sp.gran, sp.start) for s, e, sp in extract_spans(text, drop=drop)]


def roles(text: str) -> dict[str, str | None]:
    """``{texto de la expresión: papel}`` (``"citation"``, ``"metadata"`` o ``None``)."""
    return {text[s:e]: role for s, e, _, role in extract_spans_with_role(text)}


# ----------------------------------------------------------------------------- fechas correctas (no deben romperse)

def test_real_dates_still_found():
    """Las fechas reales de todos los formatos se siguen reconociendo tras endurecer las reglas."""
    # texto → (expresión esperada, granularidad esperada)
    cases = {
        "In the first week of October 1942, Hitler came to recognise": ("October 1942", "month"),
        "attorney general 1866 to 10 March 1880, and bailiff": ("10 March 1880", "day"),
        "the anti-Semitic Nuremberg Race Laws in the 1930s.": ("the 1930s", "decade"),
        "from the Middle East during the early 15th century, but": ("early 15th century", "century"),
        "in the period from 1989 to 1993.": ("from 1989 to 1993", "range"),
        "In 1999, German SPD Minister of Defence Rudolf Scharping": ("1999", "year"),
        "sexual abstinence from the fourth century on": ("fourth century", "century"),
        "has stayed relatively stable since mid-2021 – meaning": ("2021", "year"),
        "Soviet art 1920 was dominated by the avant-garde": ("1920", "year"),   # "art" sin punto no es "art."
        "On Dec 6, 2013, two software engineers": ("Dec 6, 2013", "day"),
    }
    for text, (surface, gran) in cases.items():
        found = [(t, g) for t, g, _ in spans(text)]
        assert (surface, gran) in found, f"{text!r}: se esperaba {surface!r} ({gran}), salió {found}"


def test_compound_ordinal_century():
    """"twenty first century" (sin guion) es el siglo XXI, no el I (error visto en TT0)."""
    for text in ("in the twenty first century", "in the twenty-first century", "the Twenty First Century"):
        found = spans(text)
        assert len(found) == 1 and found[0][1:] == ("century", 2000), f"{text!r}: salió {found}"
    # El ordinal simple sigue funcionando: "first century" es el siglo I.
    assert spans("in the first century AD")[0][1:] == ("century", 0)


def test_coarse_forms_k4():
    """K4: década y siglo de las fechas, sin repetir lo que ya está en el texto."""
    cases = {
        "What happened in 1834 in Ohio?": ["1830s", "19th century"],
        "the 1830s banking crisis": ["19th century"],
        "nineteenth century novels": ["19th century"],          # forma numérica del siglo
        "During the 19th century in 1834": ["1830s"],           # el siglo ya está
        "In 300 BC Rome": ["3rd century BC"],
        "the war of 1812-1815": ["1810s", "19th century"],
        "Bitcoin price in 2017": ["2010s"],                     # sin "21st century"
        "the first years, 1801 and 1805": ["19th century"],     # "1800s" sería el siglo, no la década
        "currently the best": [],
    }
    for text, expected in cases.items():
        assert coarse_forms(text) == expected, f"{text!r}: se esperaba {expected}, salió {coarse_forms(text)}"


# ----------------------------------------------------------------------------- números que no son fechas

def test_no_dates_inside_urls():
    """Los números dentro de URLs y BBCode no son fechas; fuera de la URL, sí."""
    assert spans("[url=http://syw79.com/home.php?mod=space&uid=1738]sewgmf Whether") == []
    assert spans("idnesautoelectrical.co.uk/renault-trafic-dci-2004/#comment-63901]cnmkdj only") == []
    assert spans("see www.example.org/archive/1999/report for the 1834 reform") == [("1834", "year", 1834)]


def test_numbers_with_page_section_money_context():
    """Números precedidos de página, sección, formulario o símbolo de moneda, o seguidos de unidad, no son años."""
    assert spans("1464(t) (applicable to Federal savings associations)") == []
    assert spans("help alleviate the burden on ( printed page 1654) individuals") == []
    assert spans("It costs $1200-1500 to advertise for bids.") == []           # ni rango ni "1500" suelto
    assert spans("Report it on Form 1099 by January") == []
    assert spans("See Section 1983 claims") == []
    assert spans("He wrote 1500 words") == []
    assert spans("2d 1284, 1307-08 (M.D. Pa.)") == []                         # página de "F.2d"


# ----------------------------------------------------------------------------- papel: cita bibliográfica

def test_citation_years():
    """Años de referencias bibliográficas y judiciales: papel ``citation``."""
    # (texto, año que debe salir como cita)
    cites = [
        ("on in English Politics 1820 to 1830 (Oxford, 1964), 113–15.", "1964"),
        ("(for a different context, see (Wang et al., 2016 How time preferences differ", "2016"),
        ("Poterba, J.M. & Venti, S.F. & Wise, D.A., 1992.", "1992"),
        ("including Frankel and Lee’s (1998) intrinsic-to-market value", "1998"),
        ("f attack to the second theorem (see Feferman 1960).", "1960"),
        ("and: Origins of Military Rule (London: Z ed, 1978), p.", "1978"),
        ("erica (Ithaca, NY: Cornell University Press, 2004), 148.", "2004"),
        ("encrypting it (Head 2019; Mandiant 2016), and engaging", "2019"),
        ("encrypting it (Head 2019; Mandiant 2016), and engaging", "2016"),
        ("305 AD2d 28, 760 NYS2d 431 (1st Dept 2003) – “This appeal", "2003"),
        ("An Australian Study\" [2014] UNSWLawJl 39;", "2014"),
        ("(2014) 37(3) UNSW Law Journal 1062", "2014"),
        ("(2017).", "2017"),
        ("184. a b Long 2005, p.", "2005"),                                      # año + página, sin paréntesis
        ("ICRC, *Annual Report* 1975, p.", "1975"),
        ("Bogna Wilczyńska Studia Judaica, Nr 2 (36), 2015, pp.", "2015"),
        ("25(3), 505–512 (1995) 21.", "1995"),                                    # volumen y páginas + (año)
        ("PLoS ONE 8, e61912 (2013).", "2013"),
        ("2nd ed, 2022.", "2022"),                                                # edición / volúmenes
        ("biography with letters), 4 vols, 1516; as Patristic Scholarship", "1516"),
        ("Boutroux 1908).", "1908"),
        ("Peacock, Darren (2002), ‘Statistics, structure and satisfaction’", "2002"),  # entrada de bibliografía
        ("Toronto: University of Toronto Press, 2011.", "2011"),                  # pie de imprenta sin paréntesis
        ("Den Haag: Martinus Nijhoff, 1952; *Ideas Pertaining", "1952"),
        ("Reuben Fine, The Psychology of the Chess Player (Dover Pub., 1956);", "1956"),
        ("His Fast-Day Sermon (Prince Society, Boston, 1876) The Andros Tracts", "1876"),
        ("public (Blinder, Goodhart, Hildebrand, Lipton, & Wyplosz, 2001).", "2001"),
        ("(Sue, Hist. du Galv., 1802, Vol. 2)", "1802"),                          # volumen / página / editor
        ("Roger Lowenstein, *America’s Bank* (2015) p. 12", "2015"),
        ("2016 edited by Maaike Warnaar, Luciano Zaccara", "2016"),
        ("Lancet. 2010; 17:1318-1325.", "2010"),                                              # "(" en el fragmento anterior
    ]
    for text, year in cites:
        r = roles(text)
        assert r.get(year) == "citation", f"{text!r}: {year} debería ser cita, roles={r}"


def test_event_years_are_not_citations():
    """Años de hechos que se parecen a citas (paréntesis, comas, titulaciones) no se marcan."""
    # (texto, año que debe ser fecha del contenido)
    events = [
        ("In 1991, the Merkle tree was used to form a secure b", "1991"),
        ("upgrades like Homestead (2016), the DAO fork (2016), Byzantium (2017)", "2017"),
        ("of Harry Carey, Sr., in 1948, 3 Godfathers (1948).", "1948"),
        ("He settled in Lyon (born in Paris, 1834) and taught", "1834"),
        ("The Origin of Species was published in 1859.", "1859"),
        ("Dr John Boden (MBBCh 1965) died in Sandton Mediclinic", "1965"),        # años de titulación
        ("Leslie John du Toit (BArch 1955) started life", "1955"),
        ("ege, England, Ph.D., 1942; Univ. of London, England, D.Sc., 1949.", "1949"),
        ("Apollo 11 (1969) landed on the Moon", "1969"),                          # número + (año) sin coma
        ("Note: The Treaty of Paris, 1783, ended the war.", "1783"),             # "Note:" no es un pie de imprenta
        ("In 1933, once the Federal Deposit Insurance Corporation began", "1933"),
        ("However, the collapse of the Bretton Woods system in 1973 and the shift", "1973"),
        ("Lieutenant, 1904; Commander, 1916; Captain, 1920; Rear-Adm., 1932;", "1916"),  # carrera, no citas
    ]
    for text, year in events:
        r = roles(text)
        assert year in r and r[year] is None, f"{text!r}: {year} debería ser fecha de contenido, roles={r}"


# ----------------------------------------------------------------------------- papel: metadatos de la página

def test_metadata_dates():
    """Fechas de la página (acceso, comentarios, copyright): papel ``metadata``."""
    # (texto, expresión que debe salir como metadato)
    meta = [
        ("lTransitionsReport-FINAL-EN.pdf (accessed 2 May 2013).", "2 May 2013"),
        ("a worn out chair 5390. **Williammig** August 4, 2025 at 7:09 pm", "August 4, 2025"),
        ("Date: April 28, 2023, 3:30 PM Place: Centre de recherches", "April 28, 2023"),
        ("© 2004 Elsevier Ltd. All rights reserved.", "2004"),
        ("Retrieved 12 March 2020 from the archive", "12 March 2020"),
    ]
    for text, surface in meta:
        r = roles(text)
        assert r.get(surface) == "metadata", f"{text!r}: {surface!r} debería ser metadato, roles={r}"


def test_drop_roles():
    """``drop`` quita exactamente los papeles pedidos y rechaza papeles desconocidos."""
    text = "Caesar crossed the Rubicon in 49 BC (Oxford, 1964). Page accessed 2 May 2013."
    assert [t for t, _, _ in spans(text)] == ["49 BC", "1964", "2 May 2013"]
    assert [t for t, _, _ in spans(text, drop=("citation",))] == ["49 BC", "2 May 2013"]
    assert [t for t, _, _ in spans(text, drop=("citation", "metadata"))] == ["49 BC"]
    try:
        extract_spans(text, drop=("bogus",))
    except ValueError:
        pass
    else:
        raise AssertionError("un papel desconocido debería dar ValueError")


if __name__ == "__main__":
    # Ejecutor mínimo sin pytest: corre todas las funciones test_* en orden alfabético.
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FALLA {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} pruebas correctas")
    sys.exit(1 if failed else 0)
