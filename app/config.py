"""Settings: defaults, overridden by config.toml, overridden by TWIN_* env vars."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "user": {"name": "Me"},
    "llm": {
        "provider": "anthropic",
        "model": "claude-opus-5",
        "live_model": "",
        "live_effort": "low",
        "deep_effort": "medium",
        "fallbacks": True,
        "ollama_url": "http://localhost:11434",
        "ollama_model": "gemma4:26b",
        "ollama_live_model": "",
        "ollama_num_ctx": 32768,
        "ollama_keep_alive": "2h",
    },
    "stt": {"backend": "auto", "model": "large-v3-turbo", "language": "", "translate": False},
    "live": {
        "suggest_interval_sec": 20,
        "min_chunk_sec": 2.5,
        "max_chunk_sec": 12,
        "silence_rms": 0.008,
        "keep_audio": False,
    },
    "server": {"host": "127.0.0.1", "port": 8765},
}


def load_settings(path: Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    path = path or Path(os.environ.get("TWIN_CONFIG", ROOT / "config.toml"))
    if path.exists():
        with open(path, "rb") as f:
            for section, values in tomllib.load(f).items():
                cfg.setdefault(section, {}).update(values)
    # e.g. TWIN_LLM_PROVIDER=ollama, TWIN_STT_BACKEND=none
    for key, val in os.environ.items():
        if not key.startswith("TWIN_") or key in ("TWIN_CONFIG", "TWIN_DATA_DIR"):
            continue
        parts = key[5:].lower().split("_", 1)
        if len(parts) == 2 and parts[0] in cfg:
            cfg[parts[0]][parts[1]] = _coerce(val, cfg[parts[0]].get(parts[1]))
    cfg["data_dir"] = Path(os.environ.get("TWIN_DATA_DIR", ROOT / "data"))
    return cfg


def _coerce(val: str, current):
    if isinstance(current, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(val)
    if isinstance(current, float):
        return float(val)
    return val
