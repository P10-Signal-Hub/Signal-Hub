import time
import json
from typing import Dict, Any
import torch
from transformers import pipeline, AutoTokenizer
import string
import re

# ---------- CONFIG ----------
MODEL_ID = "google/gemma-3-4b-it"
TASK = "text-generation"

_moderator = None
_pipeline_init_info = {"init_elapsed": None, "init_error": None}

def handle_incoming_message(username: str, message: str, debug=False):
    result = moderate_message(username, message)
    
    if debug:
        print(f"Moderation result: {result['label']} -> {result['action']}")

    # API response
    return {
        "username": result["username"],
        "message": result["message"],
        "label": result["label"],
        "confidence": result.get("confidence", None),
        "action": result["action"]
    }


def init_moderator():
    """init the model pipeline at server startup."""
    get_moderator()

# ---------- UTILITIES ----------
def _get_device_info():
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "device_name": torch.cuda.get_device_name(0),
            "available_cuda": True
        }
    else:
        return {
            "device": "cpu",
            "device_name": "CPU",
            "available_cuda": False
        }

def _init_pipeline() -> None:
    """Initialize global text-generation pipeline."""
    global _moderator, _pipeline_init_info
    if _moderator is not None:
        return

    start = time.time()
    try:
        device_info = _get_device_info()
        device_id = 0 if device_info["available_cuda"] else -1

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        _moderator = pipeline(
            task=TASK,
            model=MODEL_ID,
            tokenizer=tokenizer,
            device=device_id,
            torch_dtype="auto",
        )

        _pipeline_init_info["init_elapsed"] = time.time() - start
        _pipeline_init_info["init_error"] = None
    except Exception as e:
        _pipeline_init_info["init_elapsed"] = time.time() - start
        _pipeline_init_info["init_error"] = str(e)
        _moderator = None
        raise

def get_moderator():
    """Public accessor"""
    if _moderator is None:
        _init_pipeline()
    return _moderator

# ---------- Moderation logic ----------
def normalize_label_from_model(raw_label: str) -> str:
    """Map model label names to one of SAFE / INAPPROPRIATE / THREAT."""
    lbl = (raw_label or "").upper()
    if any(x in lbl for x in ["SAFE", "NO_HATE", "OTHER", "NEUTRAL"]):
        return "SAFE"
    if any(x in lbl for x in ["THREAT", "VIOLENCE", "HARM"]):
        return "THREAT"
    if any(x in lbl for x in ["TOXIC", "ABUSE", "INSULT", "HATE", "OFFENSIVE"]):
        return "INAPPROPRIATE"
    # default
    return "INAPPROPRIATE"

def decide_action(label: str) -> str:
    if label == "SAFE":
        return "ALLOW"
    if label == "THREAT":
        return "ESCALATE"
    return "FLAG_ADMIN"

def moderate_message(username: str, message: str) -> Dict[str, Any]:
    """
    Moderate a single message using Gemma (text-generation) with:
      - few-shot prompt,
      - robust parsing,
      - deterministic decode (no sampling flags),
      - simple heuristic fallback (keywords/regex).
    Returns same dict shape as before.
    """
    # simple heuristic classifier (fallback) tuned for speed and recall
    THREAT_WORDS = [
        r"\bkill\b", r"\bkill you\b", r"\bwatch your back\b", r"\bi will hurt\b",
        r"\bi'll hurt\b", r"\bi will kill\b", r"\bbeat you\b", r"\bshoot\b", r"\bstab\b"
    ]
    INAPPROPRIATE_WORDS = [
        r"\bnude\b", r"\bnudes\b", r"\bsex\b", r"\bwhore\b", r"\bfuck\b",
        r"\bidiot\b", r"\basshole\b", r"\bstupid\b", r"\bkill yourself\b", r"\bslut\b", r"\bkill myself\b"
    ]

    def heuristic_label(msg: str) -> str:
        m = msg.lower()
        # threat check (higher priority)
        for patt in THREAT_WORDS:
            if re.search(patt, m):
                return "THREAT"
        # inappropriate check
        for patt in INAPPROPRIATE_WORDS:
            if re.search(patt, m):
                return "INAPPROPRIATE"
        return "SAFE"

    # If pipeline fails to init, return error
    try:
        mod = get_moderator()
    except Exception as e:
        return {
            "username": username,
            "message": message,
            "label": "ERROR",
            "confidence": 0.0,
            "action": "FLAG_ADMIN",
            "raw_output": {"error": str(e)}
        }

    few_shot = (
        "You are a content moderation AI. Classify the message as exactly one word: SAFE, INAPPROPRIATE, or THREAT.\n\n"
        "SAFE: \"Thanks everyone, see you tomorrow.\" -> SAFE\n"
        "INAPPROPRIATE: \"Send me nudes now!\" -> INAPPROPRIATE\n"
        "THREAT: \"I will find you and hurt you.\" -> THREAT\n\n"
        f"Message: \"{message}\"\n"
        "Answer with exactly ONE word (SAFE, INAPPROPRIATE, or THREAT):"
    )

    try:
        out = mod(few_shot, max_new_tokens=8)
        gen_text = ""
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            gen_text = out[0].get("generated_text", "")
        elif isinstance(out, str):
            gen_text = out
        else:
            gen_text = ""

        first_token = ""
        if gen_text:
            cand = gen_text.upper()
            if "\n" in cand:
                cand_tail = cand.splitlines()[-1]
            else:
                cand_tail = cand
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
                label = "SAFE"
                confidence = 0.9
            elif first_token == "THREAT":
                label = "THREAT"
                confidence = 0.95
            else:
                label = "INAPPROPRIATE"
                confidence = 0.9
            raw_out = {"response": gen_text}
        else:
            label = heuristic_label(message)
            confidence = 0.8 if label != "SAFE" else 0.99
            raw_out = {"response": gen_text or "<no generation>", "heuristic_used": True}

        action = decide_action(label)
        return {
            "username": username,
            "message": message,
            "label": label,
            "confidence": confidence,
            "action": action,
            "raw_output": raw_out
        }

    except Exception as e:
        fallback_label = heuristic_label(message)
        fallback_conf = 0.85 if fallback_label != "SAFE" else 0.99
        return {
            "username": username,
            "message": message,
            "label": fallback_label,
            "confidence": fallback_conf,
            "action": decide_action(fallback_label),
            "raw_output": {"error": str(e), "heuristic_used": True}
        }

# ---------- Event system ----------
class EventType:
    MESSAGE = "MESSAGE"

def create_event_obj(username: str, message: str, mod_result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {}
    if mod_result.get("label") != "SAFE":
        # flagged for review / escalation
        metadata_key = "escalated" if mod_result.get("label") == "THREAT" else "flagged"
        metadata = {
            metadata_key: True,
            "mod_label": mod_result.get("label"),
            "confidence": mod_result.get("confidence")
        }
    return {
        "eventType": EventType.MESSAGE,
        "title": username,
        "description": message,
        "metadata": metadata,
    }

def save_event(event: Dict[str, Any]) -> None:
    print(f"Event saved: {event['title']} - {event['description']} | metadata={event['metadata']}")

def notify_channel_admin(event: Dict[str, Any], mod_result: Dict[str, Any]) -> None:
    print(f"Admin notified for event: {event['title']} | moderation={mod_result['label']}")

def notify_admin_urgent(event: Dict[str, Any], mod_result: Dict[str, Any]) -> None:
    print(f"URGENT: Admin alerted for event: {event['title']} | moderation={mod_result['label']}")

# ---------- Top-level batch runner that returns JSON-able dict ----------
def run_moderation_batch(messages: Dict[str, str]) -> Dict[str, Any]:
    """
    Run moderation on a batch of messages and return structured results
    """
    start_all = time.time()
    init_elapsed = None
    try:
        _init_pipeline()
        init_elapsed = _pipeline_init_info.get("init_elapsed")
    except Exception as e:
        init_elapsed = _pipeline_init_info.get("init_elapsed")

    device_info = _get_device_info()
    model_info = {
        "available_cuda": device_info["available_cuda"],
        "device": device_info["device"],
        "init_elapsed": init_elapsed,
        "model": MODEL_ID,
        "task": TASK,
    }

    events_out = []
    per_message_times = []

    for username, message in messages.items():
        t0 = time.time()
        result = moderate_message(username, message)
        t1 = time.time()

        per_message_times.append(t1 - t0)

        event_obj = create_event_obj(username, message, result)
        events_out.append(event_obj)

        if result["action"] == "ALLOW":
            save_event(event_obj)
        elif result["action"] == "BLOCK":
            print(f"Message blocked: {username} - {message}")
        elif result["action"] == "FLAG_ADMIN":
            save_event(event_obj)
            notify_channel_admin(event_obj, result)
        elif result["action"] == "ESCALATE":
            save_event(event_obj)
            notify_admin_urgent(event_obj, result)

    total_elapsed = time.time() - start_all
    avg_message_time = (sum(per_message_times) / len(per_message_times)) if per_message_times else 0.0

    results = {
        "model_information": model_info,
        "result": {
            "event": events_out,
            "event_type_statistics": {
                "event_type": EventType.MESSAGE,
                "event_type_elapsed_time": total_elapsed,
                "event_type_statistics": {
                    "avg_message_time": avg_message_time,
                    "messages_processed": len(events_out)
                }
            }
        }
    }
    return results

# ---------- CLI test execution ----------
if __name__ == "__main__":
    test_messages = {
        "alice": "Hello everyone, nice to meet you!",
        "bob": "I hate you and I will hurt you.",
        "charlie": "This is stupid, you are such an idiot.",
        "dave": "Let's grab lunch tomorrow.",
        "eve": "Send me nudes now!"
    }

    output = run_moderation_batch(test_messages)
    print("\n=== JSON OUTPUT ===")
    print(json.dumps(output, indent=4))
