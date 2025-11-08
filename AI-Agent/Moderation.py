"""
---PAYLOAD---
{
	'method': 'moderation',
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
            'event': [
                {
                    'eventType': str,
                    'title': str,
                    'description': str,
                    'metadata': { dict }
                }
            ],
            'event_type_statistics':
            {
                'event_type': str,
                'event_type_elapsed_time': float,
                'event_type_statistics': { 'avg_message_time': float, 'messages_processed': int }
            }
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
import os, time, json, re, string
from typing import Dict, Any
import torch
from transformers import pipeline, AutoTokenizer

class EventType:
    MESSAGE = "MESSAGE"

def normalize_label_from_model(raw_label: str) -> str:
    """Map model label names to one of SAFE / INAPPROPRIATE / THREAT."""
    lbl = (raw_label or "").upper()
    if any(x in lbl for x in ["SAFE", "NO_HATE", "OTHER", "NEUTRAL"]):
        return "SAFE"
    if any(x in lbl for x in ["THREAT", "VIOLENCE", "HARM"]):
        return "THREAT"
    if any(x in lbl for x in ["TOXIC", "ABUSE", "INSULT", "HATE", "OFFENSIVE"]):
        return "INAPPROPRIATE"
    return "INAPPROPRIATE"

def decide_action(label: str) -> str:
    if label == "SAFE":
        return "ALLOW"
    if label == "THREAT":
        return "ESCALATE"
    return "FLAG_ADMIN"

def heuristic_label(msg: str) -> str:
    """Simple regex-based classifier used as fallback or quick filter."""
    THREAT_WORDS = [
        r"\bkill\b", r"\bkill you\b", r"\bwatch your back\b", r"\bi will hurt\b",
        r"\bi'll hurt\b", r"\bi will kill\b", r"\bbeat you\b", r"\bshoot\b", r"\bstab\b"
    ]
    SELF_HARM_WORDS = [
        r"\bkill myself\b", r"\bkill my self\b", r"\bsuicide\b", r"\bend my life\b", r"\bself[-\s]?harm\b"
    ]
    INAPPROPRIATE_WORDS = [
        r"\bnude\b", r"\bnudes\b", r"\bsex\b", r"\bwhore\b", r"\bfuck\b",
        r"\bidiot\b", r"\basshole\b", r"\bstupid\b", r"\bslut\b"
    ]

    m = msg.lower()
    for patt in SELF_HARM_WORDS:
        if re.search(patt, m):
            return "THREAT"
    for patt in THREAT_WORDS:
        if re.search(patt, m):
            return "THREAT"
    for patt in INAPPROPRIATE_WORDS:
        if re.search(patt, m):
            return "INAPPROPRIATE"
    return "SAFE"


def moderate_message(pipe, username: str, message: str) -> Dict[str, Any]:
    """
    Use an injected HF pipeline to classify a single message.
    Falls back to a heuristic if the pipeline errors.
    """

    prompt = (
        "You are a content moderation AI. Classify the message as exactly one word: SAFE, INAPPROPRIATE, or THREAT.\n\n"
        "SAFE: \"Thanks everyone, see you tomorrow.\" -> SAFE\n"
        "INAPPROPRIATE: \"Send me nudes now!\" -> INAPPROPRIATE\n"
        "THREAT: \"I will find you and hurt you.\" -> THREAT\n\n"
        f"Message: \"{message}\"\n"
        "Answer with exactly ONE word (SAFE, INAPPROPRIATE, or THREAT):"
    )

    try:
        out = pipe(prompt, max_new_tokens=8, do_sample=False)
        gen_text = ""
        if isinstance(out, list) and out and isinstance(out[0], dict):
            gen_text = out[0].get("generated_text", "")
        elif isinstance(out, str):
            gen_text = out

        first = ""
        if gen_text:
            cand = gen_text.upper()
            tail = cand.splitlines()[-1] if "\n" in cand else cand
            tokens = [t.strip(string.punctuation) for t in tail.strip().split()]
            for t in tokens:
                if t in ("SAFE", "INAPPROPRIATE", "THREAT"):
                    first = t
                    break
            if not first:
                for label in ("SAFE", "INAPPROPRIATE", "THREAT"):
                    if label in cand:
                        first = label
                        break

        if first:
            if first == "SAFE":
                label, conf = "SAFE", 0.9
            elif first == "THREAT":
                label, conf = "THREAT", 0.95
            else:
                label, conf = "INAPPROPRIATE", 0.9
            raw_out = {"response": gen_text}
        else:
            label = heuristic_label(message)
            conf = 0.8 if label != "SAFE" else 0.99
            raw_out = {"response": gen_text or "<no generation>", "heuristic_used": True}

    except Exception as e:
        label = heuristic_label(message)
        conf = 0.85 if label != "SAFE" else 0.99
        raw_out = {"error": str(e), "heuristic_used": True}

    return {
        "username": username,
        "message": message,
        "label": label,
        "confidence": conf,
        "action": decide_action(label),
        "raw_output": raw_out
    }

def run_moderation_batch_with_pipeline(
    messages: Dict[str, str],
    pipe,
    device_type: str,
    model_name: str,
    model_task: str
) -> Dict[str, Any]:
    """
    Batch-run moderation using an injected pipeline.
    - messages: dict username -> message
    - pipe: HF pipeline instance (shared)
    """
    start_all = time.time()
    usernames = list(messages.keys())
    texts = [messages[u] for u in usernames]

    prompts = []
    for message in texts:
        prompts.append(
            "You are a content moderation AI. Classify the message as exactly one word: SAFE, INAPPROPRIATE, or THREAT.\n\n"
            "SAFE: \"Thanks everyone, see you tomorrow.\" -> SAFE\n"
            "INAPPROPRIATE: \"Send me nudes now!\" -> INAPPROPRIATE\n"
            "THREAT: \"I will find you and hurt you.\" -> THREAT\n\n"
            f"Message: \"{message}\"\n"
            "Answer with exactly ONE word (SAFE, INAPPROPRIATE, or THREAT):"
        )

    outs = []
    try:
        if pipe is None:
            raise RuntimeError("pipeline is None")
        outs = pipe(
            prompts,
            max_new_tokens=8,
            do_sample=False,
            truncation=True,
            return_full_text=False
        )
    except Exception as e:
        print(f"[moderation] batched pipeline failed: {e}; falling back to single-call loop")
        for p in prompts:
            try:
                outs.append(pipe(p, max_new_tokens=8, do_sample=False, truncation=True, return_full_text=False))
            except Exception as e2:
                outs.append(None)

    gen_texts = []
    for out in outs:
        if out is None:
            gen_texts.append("")
            continue
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            gen_texts.append(out[0].get("generated_text",""))
        elif isinstance(out, dict) and "generated_text" in out:
            gen_texts.append(out.get("generated_text",""))
        elif isinstance(out, str):
            gen_texts.append(out)
        else:
            gen_texts.append(str(out))

    events_out = []
    per_times = []
    for idx, username in enumerate(usernames):
        t0 = time.time()
        message = texts[idx]
        gen_text = gen_texts[idx]

        first_token = ""
        if gen_text:
            cand = gen_text.upper()
            cand_tail = cand.splitlines()[-1] if "\n" in cand else cand
            tokens = [t.strip(string.punctuation) for t in cand_tail.strip().split()]
            for t in tokens:
                if t in ("SAFE", "INAPPROPRIATE", "THREAT"):
                    first_token = t
                    break
            if not first_token:
                for label in ("SAFE", "INAPPROPRIATE", "THREAT"):
                    if label in cand:
                        first_token = label
                        break

        if first_token:
            if first_token == "SAFE":
                label, confidence = "SAFE", 0.9
            elif first_token == "THREAT":
                label, confidence = "THREAT", 0.95
            else:
                label, confidence = "INAPPROPRIATE", 0.9
            raw_out = {"response": gen_text}
        else:
            label = heuristic_label(message)
            confidence = 0.8 if label != "SAFE" else 0.99
            raw_out = {"response": gen_text or "<no generation>", "heuristic_used": True}

        metadata = {}
        if label != "SAFE":
            key = "escalated" if label == "THREAT" else "flagged"
            metadata = {key: True, "mod_label": label, "confidence": confidence}

        events_out.append({
            "eventType": EventType.MESSAGE,
            "title": username,
            "description": message,
            "metadata": metadata,
        })

        action = decide_action(label)
        if action == "ALLOW":
            print(f"Event saved: {username} - {message} | metadata={metadata}")
        elif action == "FLAG_ADMIN":
            print(f"Event saved: {username} - {message} | metadata={metadata}")
            print(f"Admin notified for event: {username} | moderation={label}")
        elif action == "ESCALATE":
            print(f"Event saved: {username} - {message} | metadata={metadata}")
            print(f"URGENT: Admin alerted for event: {username} | moderation={label}")

        per_times.append(time.time() - t0)

    total_elapsed = time.time() - start_all
    avg_time = (sum(per_times) / len(per_times)) if per_times else 0.0

    model_info = {
        "model_name": model_name,
        "model_task": model_task,
        "device_type": device_type,
        "elapsed_time": total_elapsed
    }
    return {
        "model_information": model_info,
        "result": {
            "event": events_out,
            "event_type_statistics": {
                "event_type": EventType.MESSAGE,
                "event_type_elapsed_time": total_elapsed,
                "event_type_statistics": {
                    "avg_message_time": avg_time,
                    "messages_processed": len(events_out)
                }
            }
        }
    }