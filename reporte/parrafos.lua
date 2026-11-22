-- Filtro de pandoc para la versión Word del reporte.
-- Los \paragraph{...} de LaTeX llegan como encabezados de nivel 4; con --number-sections
-- saldrían numerados ("2.0.0.1"). Aquí se convierten en un párrafo con el título en negrita,
-- que es como se ven en el PDF.
function Header(el)
  if el.level >= 4 then
    return pandoc.Para({pandoc.Strong(el.content)})
  end
end
