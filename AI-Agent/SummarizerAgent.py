from typing import Dict, Any, List
from Agent import Agent
from Model import Model
import Summarizer as summ

class SummarizerAgent(Agent):
    def __init__(self, model: Model):
        super().__init__(model)

    def call_agent(self, payload: Dict[str, Any]):
        return self.input_handler(payload, self._run_summarizer)

    def _run_summarizer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        conversation: List[Dict[str, Any]] = payload.get("conversation") or []

        if not conversation and isinstance(payload.get("messages"), dict):
            conversation = [
                {"type": "text", "text": f"{user}: {msg}"}
                for user, msg in payload["messages"].items()
                if msg
            ]

        if not conversation:
            return {
                "summary": "",
                "talking_points": [],
                "decisions": []
            }

        pipe = getattr(self.model, "model", None)

        try:
            if pipe is not None:
                result = summ.summarize_with_pipeline(conversation, pipe)
            else:
                result = summ._quick_heuristic(conversation)
        except Exception:
            result = summ._quick_heuristic(conversation)

        return {
            "summary": result.get("summary", ""),
            "talking_points": result.get("talking_points", []),
            "decisions": result.get("decisions", []),
        }

    def test_agent(self, payload: Dict[str, Any]):
        demo_conversation = [
            {"type": "text", "text": "Alex: Hello team, status check and next steps?"},
            {"type": "text", "text": "Sam: Homepage layout is ready for approval."},
            {"type": "text", "text": "Taylor: Let's lock in Wednesday 2:30pm review.", "event_id": "$41"},
            {"type": "text", "text": "Jordan: Agreed, I'll bring charts.", "event_id": "$42"},
        ]
        return self._run_summarizer({"conversation": demo_conversation})
