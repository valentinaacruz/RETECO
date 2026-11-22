#!/usr/bin/env bash
# Genera reporte.pdf y reporte.docx a partir de reporte.tex y referencias.bib.
#
# Requisitos: tectonic (o pdflatex + bibtex) para el PDF y pandoc >= 3.1 para el Word.
# Uso, desde esta carpeta:   ./compilar.sh            (PANDOC=/ruta/a/pandoc ./compilar.sh)
# La figura se regenera aparte con figura_dominios.py, porque necesita los runs.
set -euo pipefail
cd "$(dirname "$0")"
PANDOC="${PANDOC:-pandoc}"

# 1. PDF.
if command -v tectonic >/dev/null; then
  tectonic reporte.tex
else
  pdflatex reporte && bibtex reporte && pdflatex reporte && pdflatex reporte
fi

# 2. Word. native_numbering numera tablas y figuras en sus leyendas; parrafos.lua quita la
#    numeración de los \paragraph; citeproc formatea las citas y la bibliografía en español.
"$PANDOC" reporte.tex -f latex -t docx+native_numbering -o reporte.docx \
  --citeproc --bibliography=referencias.bib \
  -M lang=es-ES -M reference-section-title=Referencias \
  --number-sections --lua-filter=parrafos.lua --resource-path=.

# 3. En español pandoc llama "Cuadro" a las tablas; el PDF dice "Tabla" (babel es-tabla).
python3 - <<'PY'
import re, shutil, zipfile
src, tmp = "reporte.docx", "reporte.tmp.docx"
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/document.xml":
            # Tras "Cuadro" pandoc pone un espacio duro (U+00A0), que se conserva.
            data = re.sub(rb'(<w:t xml:space="preserve">)Cuadro(\xc2\xa0| )', rb"\1Tabla\2", data)
        zout.writestr(item, data)
shutil.move(tmp, src)
PY
echo "Listo: reporte.pdf y reporte.docx"
