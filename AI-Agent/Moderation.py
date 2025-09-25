# integrated_moderation.py
from typing import Dict
from transformers import pipeline
import os

# -----------------------
# Moderation logic
# -----------------------
_moderator = None

def get_moderator():
    """Lazy-init the model/pipeline so imports are cheap."""
    global _moderator
    if _moderator is None:
        _moderator = pipeline(
            "text-generation",
            model="google/gemma-3-4b-it"
        )
    return _moderator


def decide_action(label: str, username: str, message: str) -> str:
    """Decide what action to take based on moderation label."""
    if label == "SAFE":
        return "ALLOW"
    elif label == "INAPPROPRIATE":
        return "FLAG_ADMIN"
    elif label == "THREAT":
        return "ESCALATE"
    else:
        return "FLAG_ADMIN"


def moderate_message(username: str, message: str) -> Dict:
    """Return a structured moderation result."""
    mod = get_moderator()

    prompt = f"""
You are a moderation AI. Classify the following message as one of:
SAFE, INAPPROPRIATE, THREAT

Message: "{message}"

Answer with only one word.
"""
    result = mod(
        prompt,
        max_new_tokens=3,
        do_sample=False,
        return_full_text=False
    )

    raw = result[0]
    generated = raw.get("generated_text", "") if isinstance(raw, dict) else str(raw)
    text = generated.strip().upper()

    if "THREAT" in text:
        label = "THREAT"
    elif "SAFE" in text:
        label = "SAFE"
    else:
        label = "INAPPROPRIATE"

    confidence = None
    action = decide_action(label, username, message)

    return {
        "username": username,
        "message": message,
        "label": label,
        "confidence": confidence,
        "action": action,
        "raw_output": raw
    }

# -----------------------
# Event system (from program.py)
# -----------------------
class EventType:
    MESSAGE = "message"

class Event:
    def __init__(self, event_type, title, description, metadata=None):
        self.event_type = event_type
        self.title = title
        self.description = description
        self.metadata = metadata or {}

def save_event(event):
    # Implement DB/file saving here
    print(f"Event saved: {event.title} - {event.description} | metadata={event.metadata}")

def notify_channel_admin(event, mod_result):
    # Notify admin for flagged content
    print(f"Admin notified for event: {event.title} | moderation={mod_result['label']}")

def notify_admin_urgent(event, mod_result):
    # Escalate high-priority messages
    print(f"URGENT: Admin alerted for event: {event.title} | moderation={mod_result['label']}")

# -----------------------
# Message handler
# -----------------------
def handle_incoming_message(username: str, message: str):
    result = moderate_message(username, message)
    print(f"Moderation result: {result['label']} -> {result['action']}")

    if result["action"] == "ALLOW":
        ev = Event(EventType.MESSAGE, username, message)
        save_event(ev)
    elif result["action"] == "BLOCK":
        return {"status": "blocked", "reason": "violates community guidelines"}
    elif result["action"] == "FLAG_ADMIN":
        ev = Event(EventType.MESSAGE, username, message, metadata={"flagged": True, "mod_label": result["label"]})
        save_event(ev)
        notify_channel_admin(ev, result)
    elif result["action"] == "ESCALATE":
        ev = Event(EventType.MESSAGE, username, message, metadata={"escalated": True, "mod_label": result["label"]})
        save_event(ev)
        notify_admin_urgent(ev, result)

    return {"status": "processed", "moderation": result}


if __name__ == "__main__":
    # Test cases
    messages = {
        "alice": "Hello everyone, nice to meet you!",
        "bob": "I hate you and I will hurt you.",
        "charlie": "This is stupid, you are such an idiot.",
        "dave": "Let's grab lunch tomorrow.",
        "eve": "Send me nudes now!"
    }

    for user, msg in messages.items():
        print(f"\nTesting message from {user}: {msg}")
        result = handle_incoming_message(user, msg)
        print(f"Result: {result['moderation']['label']} -> {result['moderation']['action']}")
