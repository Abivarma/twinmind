"""The 'second brain': prompts that make the model reason and speak like you."""
from __future__ import annotations

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


class Twin:
    def __init__(self, llm: LLM, memory: MemoryStore, name: str):
        self.llm, self.memory, self.name = llm, memory, name

    def live_suggestions(self, segments: list[dict], title: str = "") -> dict:
        tail = transcript_text(segments, self.name, last_chars=6000)
        ctx = self.memory.context_for(title + " " + tail[-1500:])
        prompt = f"""Live conversation{f' ("{title}")' if title else ''} -- latest transcript:
<transcript>
{tail}
</transcript>

Think as {self.name}. Return ONLY a JSON object:
{{"pending_question": "question the other side just asked {self.name}, or empty",
  "say_next": "1-3 sentences {self.name} could say right now, in their voice",
  "questions_to_ask": ["up to 2 sharp questions {self.name} would ask"],
  "facts": ["up to 2 relevant facts from memory, each ending with its [source]"],
  "flags": ["up to 2 risks: commitments being made, contradictions with past positions, missing info"]}}
Use empty strings/lists when nothing is useful. Do not pad."""
        out = parse_json(self.llm.complete(_system(self.name, ctx),
                                           [{"role": "user", "content": prompt}],
                                           max_tokens=2000, live=True))
        return {k: out.get(k) or ([] if k in ("questions_to_ask", "facts", "flags") else "")
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
        full = transcript_text(segments, self.name)
        ctx = self.memory.context_for(title + " " + full[:3000] + " " + full[-3000:])
        prompt = f"""Session transcript{f' ("{title}")' if title else ''}. Lines labelled "{self.name}" are me; "Them" is everyone else (may be several people); "Note" is a private note I typed.
<transcript>
{full}
</transcript>

Return ONLY a JSON object:
{{"title": "short descriptive title",
  "summary": "markdown: 3-8 bullets covering what matters to me",
  "decisions": ["decisions made"],
  "action_items": [{{"text": "...", "owner": "me|name", "due": "date or empty"}}],
  "people": [{{"name": "Full Name", "note": "what I learned about them or what they said/want"}}],
  "projects": [{{"name": "Project", "note": "status/update from this session"}}],
  "learned_about_me": ["durable facts about ME stated by ME only: opinions, preferences, commitments, plans, background"],
  "followups": "markdown: what I should do or prepare next"}}
Only include people actually named. learned_about_me must come from my own lines, never from Them."""
        out = parse_json(self.llm.complete(_system(self.name, ctx),
                                           [{"role": "user", "content": prompt}], max_tokens=8000))
        out.setdefault("title", title or "Untitled session")
        return out

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
