import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.llm import LLM, parse_json
from app.main import create_app
from app.memory import MemoryStore
from app.stt import Chunker, Transcriber


def tone(sec, amp=0.3):
    t = np.arange(int(sec * 16000)) / 16000
    return (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype("<i2").tobytes()


def silence(sec):
    return np.zeros(int(sec * 16000), dtype="<i2").tobytes()


class FakeSTT(Transcriber):
    name = "fake"

    def __init__(self):
        self.n = 0

    def transcribe(self, audio):
        if float(np.abs(audio).max()) < 0.01:
            return {"text": "", "language": None}
        self.n += 1
        return {"text": f"utterance {self.n} about the Apollo project?", "language": "en"}

    def transcribe_file(self, path):
        return [{"start": 0.0, "text": "uploaded words"}]


class FakeLLM(LLM):
    name = "fake"

    def __init__(self):
        self.calls = []

    def complete(self, system, messages, max_tokens=4000, live=False, json_mode=False):
        prompt = messages[0]["content"] if "not valid JSON" in messages[-1]["content"] else messages[-1]["content"]
        self.calls.append((system, prompt))
        if '"say_next"' in prompt:
            return 'Sure: {"say_next": "We ship Apollo in Q3.", "questions_to_ask": ["What is the risk?"], "facts": [], "flags": [], "pending_question": "When?"}'
        if '"learned_about_me"' in prompt:
            return json.dumps({
                "title": "Apollo sync", "summary": "- Discussed Apollo",
                "decisions": ["Ship in Q3"],
                "action_items": [{"text": "Send Apollo plan", "owner": "me", "due": "Friday"}],
                "people": [{"name": "Priya Raman", "note": "Owns Apollo infra"}],
                "projects": [{"name": "Apollo", "note": "Targeting Q3"}],
                "learned_about_me": ["I prefer shipping behind feature flags"],
                "followups": "- Draft plan"})
        return "I'd say: let's ship.\nWhy: speed matters."

    def stream(self, system, messages, max_tokens=8000):
        yield "From memory: "
        yield "Apollo ships Q3 [memory/projects/apollo.md]"


@pytest.fixture
def setup(tmp_path):
    cfg = load_settings()
    cfg["data_dir"] = tmp_path
    cfg["user"]["name"] = "Abi"
    cfg["live"]["suggest_interval_sec"] = 1
    llm, stt = FakeLLM(), FakeSTT()
    return TestClient(create_app(cfg, transcriber=stt, llm=llm)), llm, tmp_path


def test_chunker_cuts_on_pause_and_drops_silence():
    c = Chunker(min_sec=1.0, max_sec=5.0, silence_rms=0.01)
    out = c.feed(silence(3))
    assert out == []  # pure silence never reaches Whisper
    out = c.feed(tone(1.5) + silence(1.0))
    assert len(out) == 1
    start, audio = out[0]
    assert audio.dtype == np.float32 and 1.5 <= len(audio) / 16000 <= 2.6
    assert c.feed(tone(6))  # max length forces a cut even without a pause


def test_parse_json_tolerant():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json("<think>{x}</think> ok {\"b\": 2}") == {"b": 2}
    assert parse_json("no json") == {}


def test_memory_search_and_extraction(tmp_path):
    m = MemoryStore(tmp_path)
    assert "profile.md" in m.list_files()
    res = m.apply_extraction({
        "learned_about_me": ["I prefer Postgres over Mongo"],
        "people": [{"name": "Priya Raman", "note": "Leads infra"}, {"name": "Them", "note": "x"}],
        "action_items": [{"text": "Email Priya", "owner": "me"}]}, "s1", "2026-09-28")
    assert "people/priya-raman.md" in res["files"] and not any("them" in f for f in res["files"])
    # idempotent for learned facts
    m.apply_extraction({"learned_about_me": ["I prefer Postgres over Mongo"]}, "s2", "2026-09-29")
    assert m.read("learned.md").count("Postgres") == 1
    assert m.search("who leads infra")[0]["path"] == "memory/people/priya-raman.md"
    assert "Postgres" in m.context_for("database choice")
    assert len(m.actions()) == 1
    with pytest.raises(ValueError):
        m.write("../../etc/passwd.md", "x")


def test_live_session_end_to_end(setup):
    client, llm, data = setup
    with client.websocket_connect("/ws/live") as ws:
        ws.send_text(json.dumps({"type": "start", "title": "Apollo sync"}))
        started = ws.receive_json()
        assert started["type"] == "started"
        sid = started["session"]
        ws.send_bytes(b"\x00" + tone(2) + silence(1))   # me
        ws.send_bytes(b"\x01" + tone(2) + silence(1))   # them (asks a question)
        ws.send_text(json.dumps({"type": "note", "text": "push for Q3"}))
        seen = {}
        while not {"segment", "suggestions"} <= seen.keys():
            m = ws.receive_json()
            seen.setdefault(m["type"], []).append(m)
        speakers = {s["speaker"] for s in seen["segment"]}
        assert {"me", "them", "note"} <= speakers or {"me", "them"} <= speakers
        assert seen["suggestions"][0]["say_next"] == "We ship Apollo in Q3."
        ws.send_text(json.dumps({"type": "answer", "text": "When do we ship?"}))
        while (m := ws.receive_json())["type"] != "answer":
            pass
        assert "Why:" in m["text"]
        ws.send_text(json.dumps({"type": "stop"}))
        while (m := ws.receive_json())["type"] != "summary":
            pass
    s = m["session"]
    assert s["summary"]["decisions"] == ["Ship in Q3"]
    assert s["title"] == "Apollo sync"
    # the twin was grounded in the profile
    assert any("My profile" in system for system, _ in llm.calls)
    # memory updated
    assert (data / "memory/people/priya-raman.md").exists()
    assert "feature flags" in (data / "memory/learned.md").read_text()
    assert (data / f"sessions/{sid}.md").exists()
    # REST views
    assert client.get("/api/sessions").json()[0]["id"] == sid
    assert client.get("/api/actions").json()[0]["text"] == "Send Apollo plan"
    assert client.get("/api/search", params={"q": "Apollo"}).json()
    # re-summarize must not duplicate action items or timeline notes
    client.post(f"/api/sessions/{sid}/summarize")
    assert len(client.get("/api/actions").json()) == 1
    assert (data / "memory/people/priya-raman.md").read_text().count("Owns Apollo") == 1


def test_chat_upload_digest_and_memory_api(setup):
    client, _, _ = setup
    r = client.post("/api/chat", json={"question": "When does Apollo ship?", "history": []})
    assert "Apollo ships Q3" in r.text and "[DONE]" in r.text
    r = client.post("/api/upload", files={"file": ("standup.m4a", b"fake", "audio/mp4")})
    assert r.json()["segments"][0]["text"] == "uploaded words"
    assert client.get("/api/digest").json()["markdown"]
    client.put("/api/memory/file", json={"path": "projects/zeus.md", "content": "# Zeus\nquantum widget"})
    assert client.get("/api/search", params={"q": "quantum"}).json()[0]["title"] == "Zeus"
    assert client.put("/api/memory/file", json={"path": "../x.md", "content": ""}).status_code == 400
    assert client.get("/api/health").json()["stt"] == "fake"


def test_ollama_live_disables_thinking_and_uses_live_model(monkeypatch):
    import httpx
    from app.llm import OllamaLLM

    sent = []

    def fake_post(url, json, timeout):
        sent.append(dict(json))
        if json.get("think") is False and len(sent) == 1:
            return httpx.Response(400, text="does not support thinking", request=httpx.Request("POST", url))
        return httpx.Response(200, json={"message": {"content": "<think>x</think>hi"}},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OllamaLLM({"ollama_url": "http://x", "ollama_model": "big", "ollama_live_model": "small"})
    assert llm.complete("sys", [{"role": "user", "content": "q"}], live=True) == "hi"
    assert sent[0]["model"] == "small" and sent[0]["think"] is False and sent[0]["keep_alive"] == "2h"
    assert "think" not in sent[1]  # retried without the toggle
    llm.complete("sys", [{"role": "user", "content": "q"}])
    assert sent[-1]["model"] == "big" and "think" not in sent[-1]


class FailingLLM(FakeLLM):
    def complete(self, system, messages, max_tokens=4000, live=False, json_mode=False):
        return self._fail("Claude not configured: set ANTHROPIC_API_KEY")


class FlakyJsonLLM(FakeLLM):
    """First summary reply is broken JSON; the retry succeeds."""

    def __init__(self):
        super().__init__()
        self.broke = False

    def complete(self, system, messages, max_tokens=4000, live=False, json_mode=False):
        if '"learned_about_me"' in messages[-1]["content"] and not self.broke:
            self.broke = True
            return '{"title": "Apollo", "summary": "- cut off'
        return super().complete(system, messages, max_tokens, live, json_mode)


def _session_with_lines(client, n=3):
    ctx = client.app.state.ctx
    s = ctx.sessions.create("Budget review")
    s["segments"] = [{"speaker": "them" if i % 2 else "me", "text": f"line {i} about the budget", "t": i}
                     for i in range(n)]
    ctx.sessions.save(s)
    return ctx, s["id"]


def test_llm_failure_keeps_transcript_and_explains(tmp_path):
    cfg = load_settings()
    cfg["data_dir"] = tmp_path
    client = TestClient(create_app(cfg, transcriber=FakeSTT(), llm=FailingLLM()))
    ctx, sid = _session_with_lines(client)
    s = client.post(f"/api/sessions/{sid}/summarize").json()
    assert s["summary_status"].startswith("failed: Claude not configured")
    assert s["summary"] is None
    md_file = tmp_path / f"sessions/{sid}.md"
    assert "line 2 about the budget" in md_file.read_text()        # transcript stored anyway
    assert "Summary status: failed" in md_file.read_text()
    r = client.get(f"/api/sessions/{sid}/export")
    assert r.status_code == 200 and "line 0 about the budget" in r.text
    assert client.get("/api/search", params={"q": "budget"}).json()  # transcript searchable


def test_bad_json_is_retried_and_actions_mirrored(tmp_path):
    cfg = load_settings()
    cfg["data_dir"] = tmp_path
    client = TestClient(create_app(cfg, transcriber=FakeSTT(), llm=FlakyJsonLLM()))
    ctx, sid = _session_with_lines(client)
    s = client.post(f"/api/sessions/{sid}/summarize").json()
    assert s["summary_status"] == "ok"
    acts = client.get("/api/actions").json()
    assert acts[0]["text"] == "Send Apollo plan" and acts[0]["session_title"] == "Budget review"
    mirror = (tmp_path / "memory/actions.md").read_text()
    assert "- [ ] Send Apollo plan — owner: me, due: Friday · Budget review" in mirror
    client.post(f"/api/actions/{acts[0]['id']}/toggle")
    assert "- [x] Send Apollo plan" in (tmp_path / "memory/actions.md").read_text()


def test_long_transcript_is_split_and_merged(tmp_path):
    from app.twin import Twin

    llm = FakeLLM()
    twin = Twin(llm, MemoryStore(tmp_path), "Abi", chunk_chars=2000)
    segs = [{"speaker": "me", "text": "x" * 300 + f" point {i}", "t": i} for i in range(20)]
    out = twin.summarize(segs, "Long call")
    extract_calls = [p for _, p in llm.calls if '"learned_about_me"' in p]
    assert len(extract_calls) > 1                       # split into parts
    assert all(len(p) < 6000 for p in extract_calls)     # each part fits the window
    assert out["action_items"] == [{"text": "Send Apollo plan", "owner": "me", "due": "Friday"}]  # deduped
    assert out["summary"]


def test_placeholder_values_are_blanked(tmp_path):
    from app.twin import Twin

    class Lazy(FakeLLM):
        def complete(self, *a, **k):
            return '{"pending_question": "empty", "say_next": "Ship it.", "questions_to_ask": ["None"], "facts": [], "flags": "none"}'
    out = Twin(Lazy(), MemoryStore(tmp_path), "Abi").live_suggestions([{"speaker": "me", "text": "hi", "t": 0}])
    assert out == {"pending_question": "", "say_next": "Ship it.", "questions_to_ask": [], "facts": [], "flags": []}
