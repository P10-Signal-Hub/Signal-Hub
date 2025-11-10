
"""
---PAYLOAD---
{
	'method': 'summarize',
	'payload':
	{
		'conversation': [ {'type': 'text', 'text': '[INPUT TEXT]'} ] #ARRAY OF CONVERSATION MESSAGES
	}
}

---RESPONSE---
{
    Response
    {
        'result':
        {
            'summary': str,
            'talking_points': [
                str,
                ...
            ],
            'decisions': [
                {
                    'text': str,
                    'source_msg_id': str | None
                },
                ...
            ]
        },
        'model_information':
        {
            'model_name': str,
            'model_task': str,
            'device_type': str,
            'elapsed_time': float
        }
    }
}
"""

import json, re
from typing import Dict, Any, List

JSON_SCHEMA_DOC = """
You are a chat summarizer for a Matrix room.

Return ONLY valid minified JSON in EXACTLY this shape:

{
  "summary": "One-line overview (max 2 sentences).",
  "talking_points": ["bullet 1", "bullet 2", "bullet 3"],
  "decisions": [
    {"text": "Decision text", "source_msg_id": "<optional-id-or-null>"}
  ]
}

Rules:
- Output JSON ONLY. No explanation, no prose, no markdown.
- All keys MUST exist.
- "talking_points" MUST be an array of strings.
- "decisions" MUST be an array (can be empty). Each item MUST have "text" and may have "source_msg_id".
- If there are no decisions, return "decisions": [].
"""

def _conversation_to_plain(conversation: List[Dict[str, Any]]) -> str:
    lines = []
    for item in conversation:
        if item.get("type") != "text":
            continue
        txt = (item.get("text") or "").strip()
        if txt:
            lines.append(txt)
    return "\n".join(lines)

def _quick_heuristic(conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = _conversation_to_plain(conversation)

    # Summary: first meaningful line, or fallback
    first_line = next((l for l in text.splitlines() if l.strip()), "")
    summary = first_line if len(first_line.split()) >= 4 else "Participants discussed updates and next steps."

    # Talking points: up to 3 medium-length, non-toxic lines
    bullets = []
    for line in text.splitlines():
        words = line.split()
        if 8 <= len(words) <= 25 and not re.search(r"\b(idiot|stupid|nude|fuck)\b", line, re.I):
            bullets.append(line.strip())
        if len(bullets) == 3:
            break
    if not bullets:
        bullets = [
            "Status updates were shared.",
            "Key topics were discussed.",
            "Next steps or follow-ups were implied."
        ]

    # Decisions: lines that sound like agreement / commitments
    decisions = []
    for item in conversation:
        if item.get("type") != "text":
            continue
        txt = (item.get("text") or "").strip()
        if re.match(r"^\s*(decided|we'?ll|let'?s|agreed|lock(ed)? in)\b", txt, re.I):
            decisions.append({
                "text": re.sub(r"^[^:]+:\s*", "", txt),
                "source_msg_id": item.get("event_id") or None
            })

    return {
        "summary": summary,
        "talking_points": bullets[:3],
        "decisions": decisions[:5]
    }

def summarize_with_pipeline(conversation: List[Dict[str, Any]], pipe) -> Dict[str, Any]:
    """
    Use Gemma (via provided `pipe`) to produce structured JSON.
    Fallback to heuristic if parsing fails.
    """
    try:
        if pipe is None:
            raise RuntimeError("No pipeline provided")

        convo_str = _conversation_to_plain(conversation)
        prompt = (
            "You are a chat summarizer for a Matrix chat log.\n"
            f"{JSON_SCHEMA_DOC}\n\n"
            "Chat log (each line 'Speaker: message'):\n"
            f"{convo_str}\n\n"
            "Now output ONLY the JSON:"
        )

        out = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            truncation=True,
            return_full_text=False
        )

        if isinstance(out, list) and out and isinstance(out[0], dict):
            gen = out[0].get("generated_text", "")
        elif isinstance(out, dict):
            gen = out.get("generated_text", "")
        else:
            gen = str(out)

        # --- DEBUG: see what the model actually returned ---
        print("[summarizer] RAW MODEL OUTPUT:", repr(gen))

        start = gen.find("{")
        end = gen.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in model output")

        candidate = gen[start:end+1].strip()
        data = json.loads(candidate)

        if not isinstance(data.get("summary"), str):
            raise ValueError("missing summary")
        if not isinstance(data.get("talking_points"), list):
            raise ValueError("missing talking_points")
        if not isinstance(data.get("decisions"), list):
            raise ValueError("missing decisions")

        norm_decisions = []
        for d in data["decisions"]:
            if isinstance(d, dict) and "text" in d:
                norm_decisions.append({
                    "text": d["text"],
                    "source_msg_id": d.get("source_msg_id")
                })
        data["decisions"] = norm_decisions

        return data
    except Exception as e:
        print(f"[summarizer] LLM failed or bad JSON, using heuristic. Error: {e}")
        return _quick_heuristic(conversation)