"""Código compartido de los experimentos de RETECO Track 1 (TEMPO).

Los módulos se separan para que lo que arranca la JVM (``bm25``, vía Pyserini) y lo
que usa torch (``dense``) nunca se carguen en el mismo proceso: en el EDA esa
combinación hacía caer el kernel.
"""
