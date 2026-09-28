"""Session persistence: JSON (source of truth) + a readable markdown copy that gets indexed."""
from __future__ import annotations

import datetime as dt
import json
import re
import threading
import uuid
import wave
from pathlib import Path

from .twin import transcript_text


class SessionStore:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "sessions"
        self.audio_dir = Path(data_dir) / "audio"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, sid: str) -> Path:
        if not re.fullmatch(r"[\w-]+", sid):
            raise ValueError("bad session id")
        return self.dir / f"{sid}.json"

    def create(self, title: str = "", kind: str = "live") -> dict:
        now = dt.datetime.now()
        sid = now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        s = {"id": sid, "title": title or f"Session {now:%b %d %H:%M}", "kind": kind,
             "started": now.isoformat(timespec="seconds"), "ended": None,
             "segments": [], "suggestions": [], "summary": None}
        self.save(s)
        return s

    def save(self, s: dict) -> None:
        with self._lock:
            tmp = self._path(s["id"]).with_suffix(".tmp")
            tmp.write_text(json.dumps(s, ensure_ascii=False, indent=1))
            tmp.replace(self._path(s["id"]))

    def get(self, sid: str) -> dict | None:
        p = self._path(sid)
        return json.loads(p.read_text()) if p.exists() else None

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.dir.glob("*.json"), reverse=True):
            s = json.loads(p.read_text())
            out.append({k: s.get(k) for k in ("id", "title", "kind", "started", "ended")} |
                       {"segments": len(s["segments"]), "summarized": bool(s.get("summary"))})
        return out

    def delete(self, sid: str) -> None:
        for p in (self._path(sid), self._path(sid).with_suffix(".md")):
            p.unlink(missing_ok=True)
        for p in self.audio_dir.glob(f"{sid}-*.wav"):
            p.unlink()

    def write_markdown(self, s: dict, name: str) -> Path:
        sm = s.get("summary") or {}
        parts = [f"# {s['title']}", f"Date: {s['started']}"]
        if sm.get("summary"):
            parts.append("## Summary\n" + sm["summary"])
        if sm.get("decisions"):
            parts.append("## Decisions\n" + "\n".join(f"- {d}" for d in sm["decisions"]))
        if sm.get("action_items"):
            parts.append("## Action items\n" + "\n".join(
                f"- [ ] {a.get('text')} ({a.get('owner') or '?'})" for a in sm["action_items"]))
        if sm.get("followups"):
            parts.append("## Follow-ups\n" + sm["followups"])
        parts.append("## Transcript\n" + transcript_text(s["segments"], name))
        p = self._path(s["id"]).with_suffix(".md")
        p.write_text("\n\n".join(parts) + "\n")
        return p


class AudioSink:
    """Optional raw audio retention (off by default, like TwinMind)."""

    def __init__(self, audio_dir: Path, sid: str, sources: list[str]):
        audio_dir.mkdir(parents=True, exist_ok=True)
        self.files = {}
        for src in sources:
            w = wave.open(str(audio_dir / f"{sid}-{src}.wav"), "wb")
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            self.files[src] = w

    def write(self, src: str, pcm: bytes) -> None:
        if src in self.files:
            self.files[src].writeframes(pcm)

    def close(self) -> None:
        for w in self.files.values():
            w.close()
