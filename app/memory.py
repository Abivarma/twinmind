"""The memory layer: human-readable markdown files + a SQLite FTS5 index over them.

data/memory/
  profile.md          who you are and how you think -- YOU own this file; never auto-edited
  learned.md          facts the twin learned from what you said (auto, dated, reviewable)
  people/<name>.md    one page per person: context + dated timeline   ("private Wikipedia")
  projects/<name>.md  one page per project
data/sessions/<id>.md readable transcript + summary per call (also indexed)
data/index.db         search index + action items
"""
from __future__ import annotations

import datetime as dt
import re
import shutil
import sqlite3
import threading
from pathlib import Path

from .config import ROOT

TEMPLATE_DIR = ROOT / "memory_template"
_STOP = set("""a an the and or but if then so to of in on at for with by from as is are was were be been
it this that these those i you he she we they me my your our their do does did not no yes can could
would should will just about what when where who why how which there here have has had""".split())


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:60] or "untitled"


class MemoryStore:
    def __init__(self, data_dir: Path):
        self.root = Path(data_dir)
        self.mem = self.root / "memory"
        self.sessions_dir = self.root / "sessions"
        for d in (self.mem / "people", self.mem / "projects", self.sessions_dir):
            d.mkdir(parents=True, exist_ok=True)
        for name in ("profile.md", "learned.md"):
            if not (self.mem / name).exists() and (TEMPLATE_DIR / name).exists():
                shutil.copy(TEMPLATE_DIR / name, self.mem / name)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.root / "index.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(path UNINDEXED, title, body);
            CREATE TABLE IF NOT EXISTS actions(
                id INTEGER PRIMARY KEY, session_id TEXT, text TEXT, owner TEXT, due TEXT,
                done INTEGER DEFAULT 0, created TEXT);
        """)
        self.reindex_all()

    # ---------- files ----------
    def _safe(self, rel: str) -> Path:
        p = (self.mem / rel).resolve()
        if self.mem.resolve() not in p.parents or p.suffix != ".md":
            raise ValueError("path must be a .md file inside the memory folder")
        return p

    def list_files(self) -> list[str]:
        return sorted(str(p.relative_to(self.mem)) for p in self.mem.rglob("*.md"))

    def read(self, rel: str) -> str:
        p = self._safe(rel)
        return p.read_text() if p.exists() else ""

    def write(self, rel: str, content: str) -> None:
        p = self._safe(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.index_file(p)

    def profile(self) -> str:
        return self.read("profile.md")

    # ---------- index ----------
    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))

    def index_file(self, p: Path) -> None:
        text = p.read_text() if p.exists() else ""
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")),
                     p.stem)
        with self._lock:
            self.db.execute("DELETE FROM chunks WHERE path = ?", (self._rel(p),))
            for body in _chunk(text):
                self.db.execute("INSERT INTO chunks(path, title, body) VALUES (?,?,?)",
                                (self._rel(p), title, body))
            self.db.commit()

    def reindex_all(self) -> None:
        with self._lock:
            self.db.execute("DELETE FROM chunks")
            self.db.commit()
        for p in list(self.mem.rglob("*.md")) + list(self.sessions_dir.glob("*.md")):
            self.index_file(p)

    def search(self, query: str, k: int = 8) -> list[dict]:
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2 and t not in _STOP]
        if not terms:
            return []
        fts = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
        with self._lock:
            rows = self.db.execute(
                "SELECT path, title, body, bm25(chunks, 0, 2.0, 1.0) AS score FROM chunks "
                "WHERE chunks MATCH ? ORDER BY score LIMIT ?", (fts, k)).fetchall()
        return [dict(r) for r in rows]

    def context_for(self, query: str, k: int = 8, budget_chars: int = 12000) -> str:
        """Everything the twin should see: profile + recent learnings + relevant memory."""
        parts = ["## My profile (authoritative)\n" + self.profile().strip()]
        learned = self.read("learned.md").strip().splitlines()
        if learned:
            parts.append("## Recently learned about me\n" + "\n".join(learned[-40:]))
        hits = [h for h in self.search(query, k)
                if h["path"] not in ("memory/profile.md", "memory/learned.md")]
        if hits:
            parts.append("## Relevant memories\n" + "\n\n".join(
                f"[{h['path']}] {h['body']}" for h in hits))
        return "\n\n".join(parts)[:budget_chars]

    # ---------- learning from a finished session ----------
    def apply_extraction(self, ex: dict, session_id: str, date: str) -> dict:
        """Merge an LLM extraction into the markdown memory. Additive only; never deletes."""
        link = f"(session {session_id})"
        changed: list[str] = []

        facts = [f.strip() for f in ex.get("learned_about_me", []) if isinstance(f, str) and f.strip()]
        if facts:
            p = self.mem / "learned.md"
            existing = p.read_text() if p.exists() else "# Learned about me\n"
            known = existing.lower()
            new = [f for f in facts if f.lower() not in known]
            if new:
                existing = existing.rstrip() + "\n" + "\n".join(f"- {date}: {f} {link}" for f in new) + "\n"
                p.write_text(existing)
                self.index_file(p)
                changed.append("learned.md")

        for kind in ("people", "projects"):
            for item in ex.get(kind, []) or []:
                name = (item.get("name") or "").strip() if isinstance(item, dict) else ""
                note = (item.get("note") or "").strip() if isinstance(item, dict) else ""
                if not name or name.lower() in ("me", "i", "them", "unknown"):
                    continue
                p = self.mem / kind / f"{slugify(name)}.md"
                if p.exists():
                    text = p.read_text().rstrip()
                else:
                    text = f"# {name}\n\n## About\n\n## Timeline"
                if "## Timeline" not in text:
                    text += "\n\n## Timeline"
                if note:
                    text += f"\n- {date}: {note} {link}"
                p.write_text(text + "\n")
                self.index_file(p)
                changed.append(f"{kind}/{p.name}")

        with self._lock:
            for a in ex.get("action_items", []) or []:
                if isinstance(a, dict) and a.get("text"):
                    self.db.execute(
                        "INSERT INTO actions(session_id, text, owner, due, created) VALUES (?,?,?,?,?)",
                        (session_id, a["text"], a.get("owner") or "", a.get("due") or "",
                         dt.datetime.now().isoformat(timespec="seconds")))
            self.db.commit()
        return {"files": changed}

    # ---------- action items ----------
    def actions(self, include_done: bool = False) -> list[dict]:
        q = "SELECT * FROM actions" + ("" if include_done else " WHERE done = 0") + " ORDER BY id DESC"
        with self._lock:
            return [dict(r) for r in self.db.execute(q).fetchall()]

    def toggle_action(self, action_id: int) -> None:
        with self._lock:
            self.db.execute("UPDATE actions SET done = 1 - done WHERE id = ?", (action_id,))
            self.db.commit()

    def delete_session_actions(self, session_id: str) -> None:
        with self._lock:
            self.db.execute("DELETE FROM actions WHERE session_id = ?", (session_id,))
            self.db.commit()


def _chunk(text: str, size: int = 900) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{p}" if cur else p
        while len(cur) > size * 2:
            out.append(cur[:size])
            cur = cur[size:]
    if cur:
        out.append(cur)
    return out
