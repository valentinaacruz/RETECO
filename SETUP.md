# Guía de configuración del entorno (RETECO – SemEval 2027, Tarea 1)

Pasos para reproducir el entorno de desarrollo en macOS. Las dependencias están fijadas en `pyproject.toml` y `uv.lock`, así que **no hace falta instalar paquetes uno por uno**: `uv sync` los instala todos con las mismas versiones.

## Requisitos previos

- macOS con Homebrew (https://brew.sh)
- Git
- ~10 GB libres (dependencias + dataset; el tamaño real del dataset depende de lo que descargue)

## 1. Obtener el proyecto

```bash
git clone <URL-DEL-REPO> RETECO
cd RETECO
```

El repositorio debe incluir `pyproject.toml` y `uv.lock`. Si no hay repo, copia esos dos archivos a una carpeta `RETECO`.

## 2. Instalar uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Abre una terminal nueva y comprueba:

```bash
uv --version
```

## 3. Instalar JDK 21 (necesario para Pyserini)

```bash
brew install openjdk@21
```

Registrarlo en el sistema (Apple Silicon; en Mac Intel cambia `/opt/homebrew` por `/usr/local`):

```bash
sudo ln -sfn /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-21.jdk
```

Configurar `JAVA_HOME`:

```bash
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 21)' >> ~/.zshrc
echo 'export PATH="$JAVA_HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verificar:

```bash
java -version
```

Debe mostrar `openjdk version "21...`.

## 4. Crear el entorno e instalar dependencias

El proyecto requiere **Python >= 3.12**; uv lo descarga si no lo tienes.

```bash
uv sync
```

Esto crea `.venv/` e instala todo lo de `pyproject.toml`, incluido el grupo `dev` (Jupyter, ruff).

Activar el entorno (opcional; también puedes anteponer `uv run` a cada comando):

```bash
source .venv/bin/activate
```

### Dependencias instaladas

| Grupo | Paquetes |
|---|---|
| Datos | `huggingface-hub`, `datasets`, `pandas`, `numpy`, `pyarrow`, `tqdm`, `scikit-learn` |
| Modelos | `torch`, `transformers`, `accelerate`, `sentence-transformers` |
| Recuperación | `faiss-cpu`, `rank-bm25`, `pyserini`, `nltk` |
| Desarrollo (`dev`) | `jupyter`, `ipykernel`, `ruff` |

Para qué sirve cada una y su documentación: ver [LIBRERIAS.md](LIBRERIAS.md).

### Paquetes opcionales (no incluidos aún)

Si usarán la API de Claude como generador o juez:

```bash
uv add anthropic python-dotenv
```

Después, guardar la clave en un archivo `.env` (nunca subirlo a git):

```
ANTHROPIC_API_KEY=...
```

Para agregar cualquier otra dependencia, usar `uv add <paquete>` (actualiza `pyproject.toml` y `uv.lock`; hay que commitear ambos).

## 5. Descargar el dataset

Dataset: https://huggingface.co/datasets/DataScience-UIBK/RETECO-SemEval2027

```bash
uv run hf download DataScience-UIBK/RETECO-SemEval2027 --repo-type dataset --local-dir ./data/RETECO-SemEval2027
```

Si el dataset exige aceptar condiciones de acceso, aceptarlas primero en la página del dataset e iniciar sesión (requiere un token personal de Hugging Face):

```bash
uv run hf auth login
```

### Verificar que la descarga está completa

```bash
uv run python -I - <<'EOF'
from pathlib import Path
from huggingface_hub import HfApi

repo = "DataScience-UIBK/RETECO-SemEval2027"
local = Path("data/RETECO-SemEval2027")
info = HfApi().dataset_info(repo, files_metadata=True)

bad = 0
for s in info.siblings:
    p = local / s.rfilename
    if not p.exists():
        print("FALTA     ", s.rfilename); bad += 1
    elif s.size is not None and p.stat().st_size != s.size:
        print("TAMAÑO MAL", s.rfilename, p.stat().st_size, "!=", s.size); bad += 1
print(f"{len(info.siblings)} archivos en remoto, {bad} problemas")
EOF
```

Debe terminar con `0 problemas`. Si falla, repetir el comando de descarga (solo baja lo que falta).

## 6. Verificar la instalación

```bash
uv run python -c "from pyserini.search.lucene import LuceneSearcher; print('pyserini ok')"
uv run python -c "import faiss, sentence_transformers, datasets, torch; print('ok, MPS:', torch.backends.mps.is_available())"
```

- Debe aparecer `pyserini ok`.
- Una advertencia de PyTorch sobre "Redirects are currently not supported in MacOs" es inofensiva.

## 7. Usar el entorno en Jupyter (opcional)

```bash
uv run python -m ipykernel install --user --name reteco --display-name "RETECO (uv)"
uv run jupyter lab
```

En el notebook, elegir el kernel **RETECO (uv)**.

## Problemas frecuentes

| Síntoma | Causa probable / solución |
|---|---|
| `UnsupportedClassVersionError` o error de `JAVA_HOME`/`libjvm` | JDK incorrecto o no registrado. Revisar `java -version` y `echo $JAVA_HOME`; abrir una terminal nueva. |
| `uv sync` falla por falta de ruedas (wheels) de algún paquete | Probar con Python 3.11 (`uv venv --python 3.11`), ajustando `requires-python` en `pyproject.toml`. |
| `401`/`403` al descargar el dataset | Falta aceptar condiciones en la página del dataset o iniciar sesión con `hf auth login`. |
| La descarga se interrumpió | Repetir el comando `hf download`; reanuda. |
| `command not found: uv` | Abrir una terminal nueva o añadir `~/.local/bin` al `PATH`. |

## Qué NO subir a git

Añadir a `.gitignore`:

```
.venv/
data/
.env
__pycache__/
```

El dataset se descarga con el paso 5; no se versiona.
