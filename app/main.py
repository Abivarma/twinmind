"""FastAPI server: live capture over WebSocket, REST for sessions/memory/chat, static UI."""
from __future__ import annotations

import asyncio
import bisect
import datetime as dt
import json
import logging
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ROOT, load_settings
from .llm import LLM, make_llm
from .memory import MemoryStore
from .sessions import AudioSink, SessionStore
from .stt import Chunker, Transcriber, make_transcriber
from .twin import Twin

log = logging.getLogger("twin")
SOURCES = {0: "me", 1: "them"}


class ChatIn(BaseModel):
    question: str
    history: list[dict] = []


class FileIn(BaseModel):
    path: str
    content: str


class Ctx:
    def __init__(self, cfg: dict, transcriber: Transcriber | None, llm: LLM | None):
        self.cfg = cfg
        self.name = cfg["user"]["name"]
        self.memory = MemoryStore(cfg["data_dir"])
        self.sessions = SessionStore(cfg["data_dir"])
        self.stt = transcriber or make_transcriber(cfg)
        self.llm = llm or make_llm(cfg)
        self.twin = Twin(self.llm, self.memory, self.name)
        self.stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._finalize_lock = threading.Lock()

    def save_transcript(self, s: dict) -> None:
        """Readable markdown copy of the session; written before (and regardless of) the LLM."""
        self.memory.index_file(self.sessions.write_markdown(s, self.name))

    def finalize(self, sid: str) -> dict | None:
        """Summarize a session and fold what was learned into memory (blocking).

        The transcript is persisted first, so an LLM failure never loses it; a failed summary
        keeps any earlier good summary and records why it failed.
        """
        with self._finalize_lock:
            s = self.sessions.get(sid)
            if not s:
                return None
            self.save_transcript(s)
            if not s["segments"]:
                s["summary_status"] = "empty: nothing was transcribed (check the STT backend and mic)"
                self.sessions.save(s)
                self.save_transcript(s)
                return s
            s["summary_status"] = "running"
            self.sessions.save(s)
            summary = self.twin.summarize(s["segments"], s["title"])
            if "_error" in summary:
                s["summary_status"] = "failed: " + summary["_error"]
                self.sessions.save(s)
                self.save_transcript(s)
                return s
            s["summary_status"] = "ok" + (f" ({summary.pop('_warning')})" if "_warning" in summary else "")
            s["summary"] = summary
            if summary.get("title") and s["title"].startswith("Session "):
                s["title"] = summary["title"]
            self.memory.delete_session_actions(sid)
            to_apply = summary if not s.get("memory_applied") else {
                "action_items": summary.get("action_items", [])}
            s["memory_update"] = self.memory.apply_extraction(
                to_apply, sid, s["started"][:10], session_title=s["title"])
            s["memory_applied"] = True
            self.sessions.save(s)
            self.save_transcript(s)
            return s


class LiveRun:
    """One live capture: audio in -> chunks -> transcript -> periodic twin suggestions."""

    def __init__(self, ctx: Ctx, ws: WebSocket, session: dict):
        self.ctx, self.ws, self.s = ctx, ws, session
        lc = ctx.cfg["live"]
        self.chunkers = {src: Chunker(lc["min_chunk_sec"], lc["max_chunk_sec"], lc["silence_rms"])
                         for src in SOURCES.values()}
        self.sink = (AudioSink(ctx.sessions.audio_dir, session["id"], list(SOURCES.values()))
                     if lc["keep_audio"] else None)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.dirty = False
        self.question = asyncio.Event()
        self.busy = False
        self.loop = asyncio.get_running_loop()
        self.tasks = [asyncio.create_task(self.stt_worker()), asyncio.create_task(self.twin_loop())]

    async def send(self, msg: dict) -> None:
        try:
            await self.ws.send_json(msg)
        except Exception:  # client went away; keep processing
            pass

    async def on_audio(self, data: bytes) -> None:
        src = SOURCES.get(data[0])
        if not src or len(data) < 3:
            return
        pcm = data[1:] if len(data) % 2 == 1 else data[1:-1]
        if self.sink:
            self.sink.write(src, pcm)
        for start, audio in self.chunkers[src].feed(pcm):
            await self.queue.put((src, start, audio))

    async def stt_worker(self) -> None:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            src, start, audio = item
            t0 = time.time()
            r = await self.loop.run_in_executor(self.ctx.stt_pool, self.ctx.stt.transcribe, audio)
            if not r["text"]:
                continue
            seg = {"speaker": src, "text": r["text"], "t": round(start, 2), "lang": r.get("language")}
            self.add_segment(seg)
            await self.send({"type": "segment", **seg, "latency": round(time.time() - t0, 2)})
            if src == "them" and "?" in r["text"]:
                self.question.set()

    def add_segment(self, seg: dict) -> None:
        segs = self.s["segments"]
        bisect.insort(segs, seg, key=lambda x: x["t"])
        self.dirty = True
        self.ctx.sessions.save(self.s)

    async def twin_loop(self) -> None:
        interval = self.ctx.cfg["live"]["suggest_interval_sec"]
        while True:
            try:
                await asyncio.wait_for(self.question.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            self.question.clear()
            if self.dirty and not self.busy and self.s["segments"]:
                await self.suggest()

    async def suggest(self) -> None:
        self.busy, self.dirty = True, False
        await self.send({"type": "thinking"})
        try:
            out = await self.loop.run_in_executor(
                None, self.ctx.twin.live_suggestions, list(self.s["segments"]), self.s["title"])
            out["t"] = self.s["segments"][-1]["t"] if self.s["segments"] else 0
            self.s["suggestions"].append(out)
            await self.send({"type": "suggestions", **out})
        finally:
            self.busy = False

    async def answer(self, question: str) -> None:
        await self.send({"type": "thinking"})
        text = await self.loop.run_in_executor(
            None, self.ctx.twin.answer_as_me, list(self.s["segments"]), question)
        await self.send({"type": "answer", "question": question, "text": text or "(no answer - check LLM settings)"})

    async def stop(self) -> None:
        for src, ch in self.chunkers.items():
            for start, audio in ch.flush():
                await self.queue.put((src, start, audio))
        await self.queue.put(None)
        await self.tasks[0]
        self.tasks[1].cancel()
        if self.sink:
            self.sink.close()
        self.s["ended"] = dt.datetime.now().isoformat(timespec="seconds")
        self.ctx.sessions.save(self.s)
        self.ctx.save_transcript(self.s)


def create_app(cfg: dict | None = None, transcriber: Transcriber | None = None,
               llm: LLM | None = None) -> FastAPI:
    cfg = cfg or load_settings()
    ctx = Ctx(cfg, transcriber, llm)
    app = FastAPI(title="TwinMind Local")
    app.state.ctx = ctx
    threading.Thread(target=lambda: ctx.stt.transcribe(np.zeros(16000, dtype=np.float32)),
                     daemon=True).start()  # warm up / download the Whisper model

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/api/health")
    def health():
        llm_cfg = cfg["llm"]
        model = llm_cfg["ollama_model"] if ctx.llm.name == "ollama" else llm_cfg["model"]
        return {"stt": ctx.stt.name, "stt_model": cfg["stt"]["model"], "llm": ctx.llm.name,
                "model": model, "name": ctx.name,
                "private": ctx.llm.name == "ollama" and ctx.stt.name != "none",
                "keep_audio": cfg["live"]["keep_audio"]}

    # ---------- live capture ----------
    @app.websocket("/ws/live")
    async def live(ws: WebSocket):
        await ws.accept()
        run: LiveRun | None = None
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if run:
                        await run.on_audio(msg["bytes"])
                    continue
                data = json.loads(msg.get("text") or "{}")
                kind = data.get("type")
                if kind == "start" and not run:
                    run = LiveRun(ctx, ws, ctx.sessions.create(data.get("title", "")))
                    await ws.send_json({"type": "started", "session": run.s["id"], "title": run.s["title"]})
                elif kind == "note" and run and data.get("text"):
                    t = max(c.samples_seen for c in run.chunkers.values()) / 16000
                    seg = {"speaker": "note", "text": data["text"], "t": round(t, 2)}
                    run.add_segment(seg)
                    await ws.send_json({"type": "segment", **seg})
                elif kind == "suggest" and run and not run.busy:
                    run.dirty = True
                    asyncio.create_task(run.suggest())
                elif kind == "answer" and run:
                    asyncio.create_task(run.answer(data.get("text", "")))
                elif kind == "stop" and run:
                    await run.stop()
                    await ws.send_json({"type": "stopped", "session": run.s["id"]})
                    sid, run = run.s["id"], None
                    await ws.send_json({"type": "status", "text": "Summarizing and updating memory..."})
                    s = await asyncio.get_running_loop().run_in_executor(None, ctx.finalize, sid)
                    await ws.send_json({"type": "summary", "session": s})
        except WebSocketDisconnect:
            pass
        finally:
            if run:  # browser closed mid-call: keep what we have
                await run.stop()
                asyncio.get_running_loop().run_in_executor(None, ctx.finalize, run.s["id"])

    # ---------- sessions ----------
    @app.get("/api/sessions")
    def sessions():
        return ctx.sessions.list()

    @app.get("/api/sessions/{sid}")
    def session(sid: str):
        s = ctx.sessions.get(sid)
        if not s:
            raise HTTPException(404)
        return s

    @app.get("/api/sessions/{sid}/export")
    def export(sid: str):
        s = ctx.sessions.get(sid)
        if not s:
            raise HTTPException(404)
        path = ctx.sessions.write_markdown(s, ctx.name)
        return FileResponse(path, media_type="text/markdown", filename=f"{s['title'][:60]} ({sid}).md")

    @app.post("/api/sessions/{sid}/summarize")
    def resummarize(sid: str):
        if not ctx.sessions.get(sid):
            raise HTTPException(404)
        return ctx.finalize(sid)

    @app.delete("/api/sessions/{sid}")
    def delete_session(sid: str):
        ctx.sessions.delete(sid)
        ctx.memory.delete_session_actions(sid)
        ctx.memory.write_actions_md()
        ctx.memory.reindex_all()
        return {"ok": True}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "audio.wav").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
        loop = asyncio.get_running_loop()
        try:
            segs = await loop.run_in_executor(ctx.stt_pool, ctx.stt.transcribe_file, tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        s = ctx.sessions.create(Path(file.filename or "Upload").stem, kind="upload")
        s["segments"] = [{"speaker": "them", "text": x["text"], "t": round(x["start"], 2)}
                         for x in segs if x["text"]]
        s["ended"] = dt.datetime.now().isoformat(timespec="seconds")
        ctx.sessions.save(s)
        return await loop.run_in_executor(None, ctx.finalize, s["id"])

    @app.get("/api/paths")
    def paths():
        d = Path(cfg["data_dir"]).resolve()
        return {"sessions": str(d / "sessions"), "memory": str(d / "memory"),
                "actions": str(d / "memory" / "actions.md")}

    # ---------- chat over memory ----------
    @app.post("/api/chat")
    def chat(body: ChatIn):
        def gen():
            for piece in ctx.twin.chat_stream(body.question, body.history):
                yield f"data: {json.dumps({'text': piece})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---------- memory ----------
    @app.get("/api/memory/files")
    def memory_files():
        return ctx.memory.list_files()

    @app.get("/api/memory/file")
    def memory_read(path: str):
        try:
            return {"path": path, "content": ctx.memory.read(path)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.put("/api/memory/file")
    def memory_write(body: FileIn):
        try:
            ctx.memory.write(body.path, body.content)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.get("/api/search")
    def search(q: str):
        return ctx.memory.search(q, 20)

    # ---------- actions & digest ----------
    @app.get("/api/actions")
    def actions(all: bool = False):
        return ctx.memory.actions(include_done=all)

    @app.post("/api/actions/{aid}/toggle")
    def toggle(aid: int):
        ctx.memory.toggle_action(aid)
        return {"ok": True}

    @app.get("/api/digest")
    def digest(date: str = ""):
        date = date or dt.date.today().isoformat()
        day = [ctx.sessions.get(x["id"]) for x in ctx.sessions.list() if x["started"].startswith(date)]
        return {"date": date, "markdown": ctx.twin.digest(day, ctx.memory.actions(), date)}

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


def run() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_settings()
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    print(f"\n  TwinMind Local -> http://{host}:{port}\n")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()
