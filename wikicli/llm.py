"""Local Gemma calls through the Ollama HTTP API (http://127.0.0.1:11434).

Only the standard library is used, so the harness has no Python dependencies.
The model never reads files itself: every call receives exactly the messages
the harness assembles.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict)


class OllamaClient:
    def __init__(self, host: str, model: str, num_ctx: int, timeout: float = 900, think: bool = False):
        self.think = think
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.timeout = timeout

    # --- low-level -----------------------------------------------------------
    def _open(self, path: str, payload: dict | None = None, timeout: float | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise LLMError(f"Ollama returned HTTP {e.code} for {path}: {body}") from e
        except (urllib.error.URLError, ConnectionError, socket.timeout) as e:
            raise LLMError(
                f"Cannot reach the local model runtime at {self.host} ({e}).\n"
                "Start Ollama (open the Ollama app, or run `ollama serve`) and try again."
            ) from e

    def _get(self, path: str) -> dict:
        with self._open(path, timeout=5) as r:
            return json.loads(r.read())

    # --- checks --------------------------------------------------------------
    def version(self) -> str:
        return self._get("/api/version").get("version", "unknown")

    def local_models(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def check(self) -> None:
        names = self.local_models()
        wanted = {self.model, self.model + ":latest"}
        if not wanted & set(names):
            have = ", ".join(names) or "none"
            raise LLMError(
                f"Model '{self.model}' is not downloaded (local models: {have}).\n"
                f"While online, run: ollama pull {self.model}\n"
                "Or choose another local model with --model or wiki.json."
            )

    def memory_snapshot(self) -> dict:
        """Loaded-model size reported by Ollama plus resident memory of Ollama processes."""
        snap: dict = {}
        try:
            for m in self._get("/api/ps").get("models", []):
                if m.get("name") in (self.model, self.model + ":latest"):
                    snap["model_size_bytes"] = m.get("size")
                    snap["model_size_vram_bytes"] = m.get("size_vram")
                    snap["context_length"] = m.get("context_length")
        except LLMError:
            pass
        snap["ollama_rss_bytes"] = ollama_rss_bytes()
        return snap

    # --- generation ----------------------------------------------------------
    def chat(self, messages: list[dict], temperature: float = 0.2, json_mode: bool = False,
             on_token=None) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think,          # Gemma 4 thinking mode: off by default for speed on small devices
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"
        start = time.perf_counter()
        parts: list[str] = []
        final: dict = {}
        with self._open("/api/chat", payload) as r:
            for line in r:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise LLMError(f"Model error: {chunk['error']}")
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
                if chunk.get("done"):
                    final = chunk
        return LLMResult(
            text="".join(parts).strip(),
            seconds=round(time.perf_counter() - start, 2),
            prompt_tokens=final.get("prompt_eval_count", 0),
            output_tokens=final.get("eval_count", 0),
            raw={k: v for k, v in final.items() if k != "message"},
        )


def ollama_rss_bytes() -> int | None:
    try:
        out = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total = sum(int(line.split(None, 1)[0]) for line in out.splitlines()
                if "ollama" in line.lower() and line.strip())
    return total * 1024 if total else None


def internet_reachable(timeout: float = 1.5) -> bool:
    """Best-effort check used to label evidence as online/offline."""
    for host in ("1.1.1.1", "8.8.8.8"):
        try:
            with socket.create_connection((host, 53), timeout=timeout):
                return True
        except OSError:
            continue
    return False
