"""Cliente LLM con caché en disco (PLAN_EXPERIMENTOS.md §1 y §7).

Cada generación se guarda en ``outputs/llm_cache/<prompt_id>.jsonl`` con clave
``sha1(modelo, system, prompt, max_tokens)``: nada se genera dos veces, y los textos
quedan disponibles para declararlos en el *system description*.

Backends:
- ``mlx``: modelo local con ``mlx-lm`` (hay que instalarlo: ``uv add mlx-lm``). Solo
  funciona en Apple Silicon (el Mac), no en el servidor.
- ``openai``: cualquier servidor compatible con la API de OpenAI (LM Studio, Ollama,
  vLLM, el endpoint compatible de Gemini). La URL va en ``base_url`` o en
  ``OPENAI_BASE_URL``, y la clave en ``OPENAI_API_KEY``.

La decodificación es voraz (temperatura 0) para que sea reproducible.

Uso típico::

    llm = LLM("mlx")
    texto = llm.generate("tg_extract_v1", doc_id, prompt, system=SYSTEM)
"""

import hashlib
import json
import os
import re
from pathlib import Path

from reteco.data import ROOT, SEED

CACHE_DIR = ROOT / "outputs" / "llm_cache"
# Modelo por defecto de cada backend: Qwen3 4B instruct (cabe en los 16 GB del M2 Pro).
DEFAULT_MODELS = {
    "mlx": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "openai": "qwen3-4b-instruct-2507",
}
# Bloques de razonamiento "<think>...</think>" de los modelos con thinking: se quitan de la salida.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class LLM:
    """Generador de texto con caché por ``prompt_id``.

    Los modelos y clientes se cargan la primera vez que se usan, así que crear un
    ``LLM`` es barato aunque todo esté ya en caché.

    Attributes:
        backend: ``"mlx"`` u ``"openai"``.
        max_tokens: tokens máximos por respuesta (forma parte de la clave de caché).
        model_name: id del modelo.
        base_url: URL del servidor compatible con OpenAI (solo backend ``openai``).
    """

    def __init__(self, backend: str = "mlx", model: str | None = None, max_tokens: int = 512,
                 base_url: str | None = None):
        """Configura el cliente sin cargar nada todavía.

        Args:
            backend: ``"mlx"`` u ``"openai"``.
            model: id del modelo; por defecto el de ``DEFAULT_MODELS``.
            max_tokens: tokens máximos por respuesta.
            base_url: URL del servidor OpenAI-compatible; por defecto ``$OPENAI_BASE_URL``.
        """
        self.backend, self.max_tokens = backend, max_tokens
        self.model_name = model or DEFAULT_MODELS[backend]
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._model = self._tokenizer = self._client = None   # se cargan bajo demanda
        self._caches: dict[str, dict[str, str]] = {}          # {prompt_id: {clave: salida}} en memoria

    # ------------------------------------------------------------------ caché

    def _cache_path(self, prompt_id: str) -> Path:
        """Archivo de caché de un prompt: ``outputs/llm_cache/<prompt_id>.jsonl``."""
        return CACHE_DIR / f"{prompt_id}.jsonl"

    def _cache(self, prompt_id: str) -> dict[str, str]:
        """Caché en memoria de un prompt; la primera vez la lee del archivo.

        Returns:
            ``{clave: salida}``. Se modifica en sitio al generar textos nuevos.
        """
        if prompt_id not in self._caches:
            cache = {}
            path = self._cache_path(prompt_id)
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            cache[r["key"]] = r["output"]
            self._caches[prompt_id] = cache
        return self._caches[prompt_id]

    def _key(self, prompt: str, system: str | None) -> str:
        """Clave de caché: sha1 de modelo, system, prompt y max_tokens.

        Cambiar cualquiera de los cuatro produce una generación nueva; ``item_id`` no
        entra, así que dos elementos con el mismo prompt comparten salida.
        """
        raw = json.dumps([self.model_name, system or "", prompt, self.max_tokens], ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def cached(self, prompt_id: str, prompt: str, system: str | None = None) -> bool:
        """``True`` si la respuesta a este prompt ya está en caché (sirve para contar lo pendiente)."""
        return self._key(prompt, system) in self._cache(prompt_id)

    # ------------------------------------------------------------------ generación

    def generate(self, prompt_id: str, item_id: str, prompt: str, system: str | None = None) -> str:
        """Texto generado para ``prompt``; lo lee de la caché si ya existe.

        Args:
            prompt_id: nombre del prompt (archivo de caché), p. ej. ``"tg_extract_v1"``.
            item_id: id del elemento (consulta, documento); solo se guarda para trazabilidad.
            prompt: mensaje del usuario.
            system: mensaje de sistema opcional.

        Returns:
            La respuesta del modelo, sin bloques ``<think>`` y sin espacios en los extremos.

        Side effects:
            Si la respuesta es nueva, añade una línea al archivo de caché del prompt.
        """
        cache, key = self._cache(prompt_id), self._key(prompt, system)
        if key in cache:
            return cache[key]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        out = _THINK_RE.sub("", self._mlx(messages) if self.backend == "mlx" else self._openai(messages)).strip()
        cache[key] = out
        # Se escribe en modo "a" línea a línea: si el proceso se corta, lo generado no se pierde.
        path = self._cache_path(prompt_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "id": item_id, "model": self.model_name, "output": out},
                               ensure_ascii=False) + "\n")
        return out

    def _mlx(self, messages: list[dict]) -> str:
        """Genera con ``mlx-lm`` (carga el modelo la primera vez).

        Args:
            messages: mensajes de chat ``[{"role", "content"}]``.

        Returns:
            Texto generado (decodificación voraz, la de ``mlx_lm.generate`` por defecto).
        """
        if self._model is None:
            # Import tardío: mlx solo existe en el Mac.
            from mlx_lm import load
            self._model, self._tokenizer = load(self.model_name)
        from mlx_lm import generate
        # enable_thinking solo lo usan las plantillas de Qwen3; las demás lo ignoran.
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                     enable_thinking=False)
        return generate(self._model, self._tokenizer, prompt=prompt, max_tokens=self.max_tokens, verbose=False)

    def _openai(self, messages: list[dict]) -> str:
        """Genera con un servidor compatible con la API de OpenAI.

        Args:
            messages: mensajes de chat ``[{"role", "content"}]``.

        Returns:
            Texto generado; ``""`` si el servidor no devuelve contenido.
        """
        if self._client is None:
            from openai import OpenAI
            # Los servidores locales no piden clave, pero el cliente exige una: "local" vale.
            self._client = OpenAI(base_url=self.base_url, api_key=os.environ.get("OPENAI_API_KEY", "local"))
        resp = self._client.chat.completions.create(model=self.model_name, messages=messages, temperature=0.0,
                                                    max_tokens=self.max_tokens, seed=SEED)
        return resp.choices[0].message.content or ""


def parse_json_block(text: str):
    """Primer objeto o lista JSON de una respuesta (tolera texto o bloques ``` alrededor).

    Prueba primero una lista ``[...]`` y luego un objeto ``{...}``, tomando desde el primer
    carácter de apertura hasta el último de cierre.

    Args:
        text: respuesta del LLM.

    Returns:
        El valor JSON decodificado, o ``None`` si no hay JSON válido.
    """
    # Quita las vallas de Markdown (```json ... ```) si las hay.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    for open_c, close_c in (("[", "]"), ("{", "}")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except json.JSONDecodeError:
                continue
    return None
