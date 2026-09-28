"""The 'second brain': prompts that make the model reason and speak like you."""
from __future__ import annotations

import json
import re

from .llm import LLM, parse_json
from .memory import MemoryStore


def _system(name: str, context: str) -> str:
    return f"""You are the TwinMind of {name}: a private second brain that thinks the way {name} thinks.
Your memory of {name} is below. The profile section is authoritative; learned items and memories are supporting evidence.

Rules:
- Reason with {name}'s priorities, positions and mental models from the profile, not generic advice.
- When drafting words for {name} to say, write in first person, in {name}'s voice and style.
- Never invent facts about {name}, their company, numbers or past decisions. If memory lacks it, say so briefly.
- Respect the "Never say / never commit to" list.
- Be terse. {name} is reading this during a live conversation.

<memory>
{context}
</memory>"""


def transcript_text(segments: list[dict], name: str, last_chars: int | None = None) -> str:
    lines = []
    for s in segments:
        who = name if s["speaker"] == "me" else ("Note" if s["speaker"] == "note" else "Them")
        lines.append(f"[{_ts(s.get('t', 0))}] {who}: {s['text']}")
    text = "\n".join(lines)
    return text[-last_chars:] if last_chars else text


def _ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR, _LIST = {"type": "string"}, {"type": "array", "items": {"type": "string"}}
_NAMED = {"type": "array", "items": _obj({"name": _STR, "note": _STR})}
SUMMARY_SCHEMA = _obj({
    "title": _STR, "summary": _STR, "decisions": _LIST,
    "action_items": {"type": "array", "items": _obj({"text": _STR, "owner": _STR, "due": _STR})},
    "people": _NAMED, "projects": _NAMED, "learned_about_me": _LIST, "followups": _STR})
SUGGEST_SCHEMA = _obj({"pending_question": _STR, "say_next": _STR, "questions_to_ask": _LIST,
                       "facts": _LIST, "flags": _LIST})
MERGE_SCHEMA = _obj({"title": _STR, "summary": _STR, "followups": _STR})

_NONE = {"", "empty", "none", "n/a", "na", "null", "no", "-", "nothing"}


def _clean(value, is_list: bool):
    """Small models often write 'empty'/'none' instead of leaving a field blank."""
    if is_list:
        items = value if isinstance(value, list) else []
        return [str(x) for x in items if str(x).strip().lower().strip(".") not in _NONE]
    text = value if isinstance(value, str) else ""
    return "" if text.strip().lower().strip(".") in _NONE else text


def _split_segments(segments: list[dict], name: str, limit: int) -> list[str]:
    parts, cur = [], []
    for seg in segments:
        cur.append(seg)
        if len(transcript_text(cur, name)) > limit and len(cur) > 1:
            parts.append(transcript_text(cur[:-1], name))
            cur = [seg]
    if cur:
        parts.append(transcript_text(cur, name))
    return parts


class Twin:
    def __init__(self, llm: LLM, memory: MemoryStore, name: str, chunk_chars: int | None = None):
        self.llm, self.memory, self.name = llm, memory, name
        # ~4 chars/token. Ollama's 32K window must also hold memory + output; Claude has 1M.
        self.chunk_chars = chunk_chars or (60_000 if llm.name == "ollama" else 600_000)

    def live_suggestions(self, segments: list[dict], title: str = "") -> dict:
        tail = transcript_text(segments, self.name, last_chars=6000)
        ctx = self.memory.context_for(title + " " + tail[-1500:])
        prompt = f"""Live conversation{f' ("{title}")' if title else ''} -- latest transcript:
<transcript>
{tail}
</transcript>

Think as {self.name}. Return ONLY a JSON object:
{{"pending_question": "question the other side just asked {self.name}, or \"\" if none",
  "say_next": "1-3 sentences {self.name} could say right now, in their voice",
  "questions_to_ask": ["up to 2 sharp questions {self.name} would ask"],
  "facts": ["up to 2 relevant facts from memory, each ending with its [source]"],
  "flags": ["up to 2 risks: commitments being made, contradictions with past positions, missing info"]}}
Use empty strings/lists when nothing is useful. Do not pad."""
        out = parse_json(self.llm.complete(_system(self.name, ctx),
                                           [{"role": "user", "content": prompt}],
                                           max_tokens=2000, live=True, json_mode=SUGGEST_SCHEMA))
        return {k: _clean(out.get(k), k in ("questions_to_ask", "facts", "flags"))
                for k in ("pending_question", "say_next", "questions_to_ask", "facts", "flags")}

    def answer_as_me(self, segments: list[dict], question: str = "") -> str:
        tail = transcript_text(segments, self.name, last_chars=6000)
        ctx = self.memory.context_for((question or "") + " " + tail[-1500:])
        ask = (f'Answer this as {self.name} would: "{question}"' if question else
               f"Draft what {self.name} should say next, responding to the latest point from the other side.")
        prompt = f"""<transcript>
{tail}
</transcript>

{ask}
Write the exact words to say (first person, spoken style, max ~80 words). Then one line starting with "Why:" giving the reasoning in {self.name}'s own terms."""
        return self.llm.complete(_system(self.name, ctx), [{"role": "user", "content": prompt}],
                                 max_tokens=1500, live=True).strip()

    def summarize(self, segments: list[dict], title: str = "") -> dict:
        """Structured summary. Long calls are split so nothing falls outside the context window.

        Returns the summary dict; on failure it has "_error" set and no content keys.
        """
        parts = _split_segments(segments, self.name, self.chunk_chars)
        full = transcript_text(segments, self.name)
        ctx = self.memory.context_for(title + " " + full[:3000] + " " + full[-3000:], budget_chars=8000)
        outs = [self._extract(text, title, ctx, i, len(parts)) for i, text in enumerate(parts)]
        good = [o for o in outs if "_error" not in o]
        if not good:
            return {"_error": outs[0].get("_error", "summary failed")}
        out = good[0] if len(good) == 1 else self._merge(good, title, ctx)
        if len(good) < len(outs):
            out["_warning"] = f"{len(outs) - len(good)} of {len(outs)} transcript parts could not be summarized"
        out.setdefault("title", title or "Untitled session")
        return out

    def _extract(self, text: str, title: str, ctx: str, i: int = 0, n: int = 1) -> dict:
        part = f" (part {i + 1} of {n})" if n > 1 else ""
        prompt = f"""Session transcript{part}{f' ("{title}")' if title else ''}. Lines labelled "{self.name}" are me; "Them" is everyone else (may be several people); "Note" is a private note I typed.
<transcript>
{text}
</transcript>

Return ONLY a JSON object:
{{"title": "short descriptive title",
  "summary": "3-8 markdown bullets covering what matters to me",
  "decisions": ["decisions made"],
  "action_items": [{{"text": "...", "owner": "me|name", "due": "date or empty"}}],
  "people": [{{"name": "Full Name", "note": "what I learned about them or what they said/want"}}],
  "projects": [{{"name": "Project", "note": "status/update from this session"}}],
  "learned_about_me": ["durable facts about ME stated by ME only: opinions, preferences, commitments, plans, background"],
  "followups": "markdown bullets: what I should do or prepare next"}}
Capture EVERY action item and commitment, including small ones ("I'll send...", "can you check..."). Only include people actually named. learned_about_me must come from my own lines, never from Them."""
        msgs = [{"role": "user", "content": prompt}]
        system = _system(self.name, ctx)
        raw = self.llm.complete(system, msgs, max_tokens=12000, json_mode=SUMMARY_SCHEMA)
        out = parse_json(raw)
        if not out and raw:  # one retry: models occasionally wrap or break the JSON
            msgs += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."}]
            out = parse_json(self.llm.complete(system, msgs, max_tokens=12000, json_mode=SUMMARY_SCHEMA))
        if not out or not (out.get("summary") or out.get("action_items")):
            return {"_error": self.llm.last_error or "model did not return a valid JSON summary"}
        for key in ("summary", "followups"):  # small models echo the format hint ("markdown: ...")
            if isinstance(out.get(key), str):
                out[key] = re.sub(r"^\s*markdown( bullets)?\s*:\s*", "", out[key], flags=re.I)
        return out

    def _merge(self, outs: list[dict], title: str, ctx: str) -> dict:
        merged: dict = {"title": outs[0].get("title") or title}
        for key in ("decisions", "action_items", "people", "projects", "learned_about_me"):
            seen, items = set(), []
            for o in outs:
                for it in o.get(key) or []:
                    k = json.dumps(it, sort_keys=True).lower()
                    if k not in seen:
                        seen.add(k)
                        items.append(it)
            merged[key] = items
        parts = "\n\n".join(f"Part {i + 1}:\n{o.get('summary', '')}\nFollow-ups:\n{o.get('followups', '')}"
                             for i, o in enumerate(outs))
        combined = parse_json(self.llm.complete(_system(self.name, ctx), [{"role": "user", "content":
            f"""These are summaries of consecutive parts of one conversation:
{parts}

Return ONLY a JSON object: {{"title": "...", "summary": "3-8 markdown bullets for the whole conversation", "followups": "markdown bullets"}}"""}],
            max_tokens=4000, json_mode=MERGE_SCHEMA))
        merged["summary"] = combined.get("summary") or "\n".join(o.get("summary", "") for o in outs)
        merged["followups"] = combined.get("followups") or "\n".join(o.get("followups", "") for o in outs)
        merged["title"] = combined.get("title") or merged["title"]
        return merged

    def chat_stream(self, question: str, history: list[dict]):
        ctx = self.memory.context_for(question + " " + " ".join(h["content"] for h in history[-4:]))
        system = _system(self.name, ctx) + (
            f"\n\nYou are now in chat mode with {self.name}. Answer from memory first and cite the "
            "[memory/...] or [sessions/...] source. If memory doesn't cover it, say so, then answer "
            "from general knowledge, labelled as such.")
        msgs = [*history[-12:], {"role": "user", "content": question}]
        yield from self.llm.stream(system, msgs)

    def digest(self, sessions: list[dict], actions: list[dict], date: str) -> str:
        blob = "\n\n".join(f"### {s['title']}\n{(s.get('summary') or {}).get('summary', '')}"
                           for s in sessions) or "(no sessions)"
        acts = "\n".join(f"- {a['text']} (owner: {a['owner'] or '?'}, due: {a['due'] or '?'})"
                         for a in actions) or "(none)"
        prompt = f"""Daily digest for {date}.
Sessions:
{blob}

Open action items:
{acts}

Write a short markdown briefing for me: 1) what happened, 2) top 3 things to do tomorrow, ranked, 3) anything at risk. Be concrete."""
        return self.llm.complete(_system(self.name, self.memory.context_for(blob[:2000])),
                                 [{"role": "user", "content": prompt}], max_tokens=4000).strip()
