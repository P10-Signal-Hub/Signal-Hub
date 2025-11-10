# AI Agent Modules – Signal Hub

This project contains modular AI agents for the Signal Hub backend.  
Each agent connects to a shared Gemma model pipeline via FastAPI endpoints and follows a consistent architecture for deployment and integration.

---

## 🧠 Overview

### Agents Implemented
| Agent | Purpose | Status |
|-------|----------|--------|
| **Event Agent** | Creates event summaries and metadata for client messages. | ✅ Ready |
| **Moderation Agent** | Detects harmful, inappropriate, or threatening content. | ✅ Ready |
| **Summarizer Agent** | Produces meeting-style summaries, talking points, and decisions. | ⚙️ Working (MVP complete; container integration pending) |

---

## ⚙️ API Usage

### Base Endpoint
POST /agent/use

### Example Request – Summarizer Agent
```json
{
  "method": "summarize",
  "payload": {
    "conversation": [
      {"type": "text", "text": "Alex: Hello team, status check and next steps?"},
      {"type": "text", "text": "Sam: Homepage layout is ready for approval."},
      {"type": "text", "text": "Taylor: Let's lock in Wednesday 2:30pm review.", "event_id": "$41"},
      {"type": "text", "text": "Jordan: Agreed, I'll bring charts.", "event_id": "$42"}
    ]
  }
}
### Example Response
{
  "result": {
    "summary": "The team is checking progress and scheduling a Wednesday 2:30pm review for the homepage layout.",
    "talking_points": [
      "Homepage layout is ready for approval.",
      "Review scheduled for Wednesday 2:30pm.",
      "Jordan will bring charts."
    ],
    "decisions": [
      {"text": "Review scheduled for Wednesday 2:30pm.", "source_msg_id": "$41"},
      {"text": "Jordan will bring charts.", "source_msg_id": "$42"}
    ]
  },
  "model_information": {
    "model_name": "google/gemma-3-4b-it",
    "model_task": "text-generation",
    "device_type": "GPU"
  }
}
```
---
Further documentation regarding technical points, deployment notes, see documentation document.