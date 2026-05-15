from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import ollama

DEFAULT_MODEL = "llama3.2"
SYSTEM_PROMPT = (
    "You answer questions about a video transcript. "
    "Use only information from the transcript below. "
    "If the answer is not in the transcript, say so. "
    "Cite timestamps in [HH:MM:SS] form when relevant.\n\n"
    "TRANSCRIPT:\n{transcript}"
)
SUMMARY_PROMPT = (
    "Summarize this video transcript. Provide:\n"
    "1. A 1-2 sentence overview.\n"
    "2. 3-7 key points as bullets, each with a [HH:MM:SS] timestamp.\n"
    "3. Action items: a checklist of concrete, actionable tasks mentioned or "
    "implied in the transcript. Write each as an imperative line starting with "
    "'- [ ] ', include the [HH:MM:SS] timestamp, and name the owner if stated. "
    "If there are none, write 'None'.\n"
    "4. Any conclusions or open questions.\n"
    "Keep it under 300 words. Use only what's in the transcript.\n\n"
    "TRANSCRIPT:\n{transcript}"
)


NON_TRANSCRIPT_JSON = {"meta.json", "summary.json"}


def find_transcript_json(job_dir: Path) -> Path | None:
    """The transcript JSON is the one output file that isn't meta/summary."""
    candidates = sorted(
        p for p in job_dir.glob("*.json") if p.name not in NON_TRANSCRIPT_JSON
    )
    return candidates[0] if candidates else None


def load_transcript_segments(job_dir: Path) -> tuple[list[dict], dict]:
    """Find the JSON output for a job and return (segments, full_meta)."""
    transcript = find_transcript_json(job_dir)
    if transcript is None:
        raise FileNotFoundError(f"No transcript JSON found in {job_dir}")
    data = json.loads(transcript.read_text(encoding="utf-8"))
    return data.get("segments", []), data


def format_transcript(segments: Iterable[dict]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"[{_ts(seg['start'])}] {seg['text']}")
    return "\n".join(lines)


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# Section headers tolerate markdown bold around the number and/or label,
# e.g. "3. Action Items:", "**3. Action Items**:", "3. **Action Items**:".
_ACTION_HEADER_RE = re.compile(
    r"^\s*\**\s*(?:\d+[.)]\s*)?\**\s*action\s*items?\b", re.IGNORECASE
)
_OTHER_HEADER_RE = re.compile(
    r"^\s*\**\s*(?:\d+[.)]\s*)?\**\s*"
    r"(overview|key\s*points?|conclusions?|open\s*questions?|summary)\b",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+?)\s*$")
_CHECKBOX_RE = re.compile(r"-?\s*\[[ xX]?\]\s*")


def extract_action_items(summary_text: str) -> list[str]:
    """Pull the 'Action items' checklist out of a generated summary."""
    if not summary_text:
        return []
    items: list[str] = []
    in_section = False
    for line in summary_text.splitlines():
        if _ACTION_HEADER_RE.match(line):
            in_section = True
            continue
        if not in_section:
            continue
        if _OTHER_HEADER_RE.match(line):
            break
        m = _BULLET_RE.match(line)
        if not m:
            continue
        # Models phrase items inconsistently ("- [ ] task", "- [ts] - [ ] task");
        # drop stray checkbox tokens but keep [HH:MM:SS] timestamps.
        item = re.sub(r"\s{2,}", " ", _CHECKBOX_RE.sub(" ", m.group(1))).strip()
        if item and item.lower().rstrip(".") not in ("none", "n/a", "no action items"):
            items.append(item)
    return items


def list_models() -> list[str]:
    try:
        resp = ollama.list()
    except Exception:
        return []
    names: list[str] = []
    for m in resp.get("models", []):
        name = m.get("model") or m.get("name")
        if not name:
            continue
        if not _supports_chat(name):
            continue
        names.append(name)
    return names


def _supports_chat(name: str) -> bool:
    """Skip embedding-only models. Tries the real capability list first,
    falls back to a name heuristic if the server is too old to report it.
    """
    if "embed" in name.lower():
        return False
    try:
        info = ollama.show(name)
        caps = info.get("capabilities")
        if caps:
            return "embedding" not in caps or any(c in caps for c in ("completion", "chat"))
    except Exception:
        pass
    return True


def chat(
    transcript_text: str,
    history: list[dict],
    user_message: str,
    model: str = DEFAULT_MODEL,
) -> str:
    messages = _build_messages(transcript_text, history, user_message)
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def chat_stream(
    transcript_text: str,
    history: list[dict],
    user_message: str,
    model: str = DEFAULT_MODEL,
):
    messages = _build_messages(transcript_text, history, user_message)
    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content


def resolve_default_model() -> str | None:
    models = list_models()
    return models[0] if models else None


def summarize(transcript_text: str, model: str) -> str:
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript_text)}],
    )
    return resp["message"]["content"]


def summarize_stream(transcript_text: str, model: str):
    for chunk in ollama.chat(
        model=model,
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript_text)}],
        stream=True,
    ):
        c = chunk.get("message", {}).get("content", "")
        if c:
            yield c


def _build_messages(transcript_text: str, history: list[dict], user_message: str) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(transcript=transcript_text)},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages
