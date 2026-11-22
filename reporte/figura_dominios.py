"""Figura del reporte: nDCG@10 por dominio de los cuatro sistemas principales (train).

Lee los runs guardados en ``outputs/exp/runs/`` (baseline oficial, mejor variante BM25,
denso ``bge-base`` e híbrido BM25 + denso con BM25 a peso 0.5), los evalúa con
``reteco.evaluate`` y dibuja un panel por subtarea (1a y 1b) con un punto por sistema y
dominio. Guarda ``reporte/fig_dominios.png``.

Uso, desde la raíz del repositorio (necesita los runs, que no están en git)::

    .venv/bin/python reporte/figura_dominios.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo se guarda el archivo
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from reteco.evaluate import evaluate_variant  # noqa: E402

# (etiqueta, retriever, variante) de cada sistema, en el orden en que se presentan en el texto.
SISTEMAS = {
    "1a": [
        ("Baseline oficial", "bm25", "cruda"),
        ("BM25, mejor variante", "bm25", "rrf(limpia,titulo)"),
        ("Denso bge-base", "bge-base", "rrf(limpia,titulo)"),
        ("Híbrido BM25 + denso", "fusion", "rrf(bm25:rrf(limpia,titulo)+bge-base:rrf(limpia,titulo))w0.5,1"),
    ],
    "1b": [
        ("Baseline oficial", "bm25", "oficial"),
        ("BM25, mejor variante", "bm25", "titulo+paso_limpio"),
        ("Denso bge-base", "bge-base", "limpia+paso"),
        ("Híbrido BM25 + denso", "fusion", "rrf(bm25:titulo+paso_limpio+bge-base:limpia+paso)w0.5,1"),
    ],
}
# Paleta categórica validada (azul, naranja, aguamarina, amarillo) y una forma distinta por
# sistema, para que la identidad no dependa solo del color (impresión en gris, daltonismo).
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARCAS = ["o", "s", "^", "D"]


def tabla(subtrack: str) -> dict[str, dict[str, float]]:
    """nDCG@10 medio por dominio de cada sistema de ``subtrack``: ``{sistema: {dominio: valor}}``."""
    out = {}
    for nombre, retriever, variante in SISTEMAS[subtrack]:
        pq = evaluate_variant(retriever, variante, subtrack)
        out[nombre] = pq.groupby("domain")["ndcg_cut_10"].mean().to_dict()
    return out


def main() -> None:
    """Dibuja los dos paneles y guarda la figura."""
    datos = {s: tabla(s) for s in ("1a", "1b")}
    # Dominios ordenados por el híbrido de 1a, el mejor arriba.
    hib = datos["1a"]["Híbrido BM25 + denso"]
    dominios = sorted(hib, key=hib.get)
    y = range(len(dominios))

    fig, ejes = plt.subplots(1, 2, figsize=(9, 5.2), sharey=True)
    for ax, sub in zip(ejes, ("1a", "1b")):
        filas = datos[sub]
        # Línea gris fina del peor al mejor sistema en cada dominio: el recorrido de la mejora.
        for i, dom in enumerate(dominios):
            vals = [filas[n][dom] for n in filas]
            ax.plot([min(vals), max(vals)], [i, i], color="#c9c8c2", lw=2, zorder=1)
        for (nombre, vals), color, marca in zip(filas.items(), COLORES, MARCAS):
            ax.scatter([vals[d] for d in dominios], list(y), s=46, color=color, marker=marca,
                       edgecolor="white", linewidth=1.2, zorder=3, label=nombre)
        ax.set_title(f"Subtarea {sub}", fontsize=11, loc="left")
        ax.set_xlabel("nDCG@10 (train)")
        ax.set_xlim(0, 0.43)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        ax.grid(axis="x", color="#e6e5df", lw=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ejes[0].set_yticks(list(y), dominios)
    handles, labels = ejes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    destino = RAIZ / "reporte" / "fig_dominios.png"
    fig.savefig(destino, dpi=300)
    print(destino)


if __name__ == "__main__":
    main()
