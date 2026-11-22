"""Código compartido de los experimentos de RETECO Track 1 (TEMPO).

Módulos del paquete:

- ``data``: rutas del proyecto, cargadores de consultas, pasos, qrels y ``guidance``,
  limpieza de HTML y normalización de las etiquetas temporales.
- ``bm25``: réplica del BM25 del baseline oficial (arranca la JVM de Pyserini).
- ``dense``: encoders densos (bge, e5) con torch, codificación del corpus por trozos
  y búsqueda por producto escalar.
- ``fusion``: top-k con el desempate del oficial y Reciprocal Rank Fusion (RRF).
- ``queries``: registro de variantes de consulta de 1a y 1b (incluidas las de oráculo).
- ``evaluate``: nDCG@10 y recall con ``pytrec_eval``, macro por dominio, test pareado
  de aleatorización y registro en ``outputs/exp/resultados.csv``.
- ``temporal``: expresiones temporales con expresiones regulares, su papel (cita,
  metadato) y sus formas gruesas (década y siglo; K4).

Los módulos se separan para que lo que arranca la JVM (``bm25``, vía Pyserini) y lo
que usa torch (``dense``) nunca se carguen en el mismo proceso: en el EDA esa
combinación hacía caer el kernel.
"""
