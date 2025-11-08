from typing import Dict, Any, List
from Agent import Agent
from Model import Model
import Moderation as mod

class ModerationAgent(Agent):
    """
    Moderation agent that classifies one or more messages and returns
    results using the platform's standard Agent response envelope.
    """
    def __init__(self, model: Model):
        super().__init__(model)

    # Adapter: normalize payloads and call your module
    def _run_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts either:
          - payload['username'] + payload['message']
          - or payload['conversation'] = [{type:'text', text:'...'}, ...]
        """
        username = payload.get("username", "user")
        message = payload.get("message")

        if not message:
            convo: List[Dict[str, Any]] = payload.get("conversation", [])
            text_msgs = [m.get("text") for m in convo if m.get("type") == "text" and m.get("text")]
            message = text_msgs[-1] if text_msgs else ""

        if not message:
            info = self.model.get_model_information()
            return {
                "model_information": {**info, "elapsed_time": 0.0},
                "result": {
                    "event": [],
                    "event_type_statistics": {
                        "event_type": mod.EventType.MESSAGE,
                        "event_type_elapsed_time": 0.0,
                        "event_type_statistics": {
                            "avg_message_time": 0.0,
                            "messages_processed": 0
                        }
                    }
                }
            }

        messages = {username: message}
        return mod.run_moderation_batch_with_pipeline(
            messages=messages,
            pipe=self.model.model,  
            device_type=self.model.device_type,
            model_name=self.model.model_name,
            model_task=self.model.model_task
        )

    def _run_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts payload['messages'] as a dict of {username: message}
        """
        incoming = payload.get("messages", {})
        if not isinstance(incoming, dict) or not incoming:
            return self._run_single(payload)
        return mod.run_moderation_batch_with_pipeline(
            messages=incoming,
            pipe=self.model.model,
            device_type=self.model.device_type,
            model_name=self.model.model_name,
            model_task=self.model.model_task
        )

    def call_agent(self, payload: Dict[str, Any]):
        """
        Selects conversation vs batch vs single; returns the standard envelope.
        """
        if "conversation" in payload:
            return self._run_conversation(payload)
        if isinstance(payload.get("messages"), dict):
            return self._run_batch(payload)
        return self._run_single(payload)
    
    def _run_conversation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts payload['conversation'] = [{ 'type': 'text', 'text': 'Speaker: message', ...}, ...]
        Converts it to a {username: last_message} dict and reuses the batch path.
        """
        convo: List[Dict[str, Any]] = payload.get("conversation", [])
        messages = self._messages_from_conversation(convo, include_all=False)

        if not messages:
            info = self.model.get_model_information()
            return {
                "model_information": {**info, "elapsed_time": 0.0},
                "result": {
                    "event": [],
                    "event_type_statistics": {
                        "event_type": "MESSAGE",
                        "event_type_elapsed_time": 0.0,
                        "event_type_statistics": {
                            "avg_message_time": 0.0,
                            "messages_processed": 0
                        }
                    }
                }
            }

        return self._run_batch({"messages": messages})
    
    def _messages_from_conversation(self, convo, include_all=False):
        """
        convo: list of {'type': 'text', 'text': 'Speaker: message'}
        include_all=False => only the last message per speaker
        include_all=True  => every line, keyed as 'Speaker#1', 'Speaker#2', ...
        """
        import re
        msgs = {}
        counts = {}
        for item in convo:
            if item.get("type") != "text":
                continue
            text = item.get("text", "")
            # split "Name: message"
            m = re.match(r"^\s*([^:]+)\s*:\s*(.*)$", text)
            if m:
                speaker, content = m.group(1).strip(), m.group(2).strip()
            else:
                speaker, content = "user", text.strip()

            if include_all:
                counts[speaker] = counts.get(speaker, 0) + 1
                key = f"{speaker}#{counts[speaker]}"
                msgs[key] = content
            else:
                msgs[speaker] = content
        return msgs

    def test_agent(self, payload):
        conversations = [
            # 1) Normal / safe
            [
                {"type": "text", "text": "Alex: Hello team, how are you guys doing today?"},
                {"type": "text", "text": "Sam: Pretty good! Just finalizing the sprint tasks."},
                {"type": "text", "text": "Taylor: Same here, pushing a small bugfix before the deadline."},
                {"type": "text", "text": "Jordan: Let's meet at 3pm to review the new dashboard."},
                {"type": "text", "text": "Alex: Perfect, 3pm it is."}
            ],
            # 2) Inappropriate / insult
            [
                {"type": "text", "text": "Alex: Did you commit the latest changes?"},
                {"type": "text", "text": "Sam: I did, but you probably didn’t pull them yet."},
                {"type": "text", "text": "Taylor: Guys, let’s stay calm."},
                {"type": "text", "text": "Jordan: Sam, you’re such an idiot sometimes."},
                {"type": "text", "text": "Alex: That’s not okay. Let’s stay professional."}
            ],
            # 3) Distress / self-harm mention
            [
                {"type": "text", "text": "Alex: How’s everyone holding up this week?"},
                {"type": "text", "text": "Sam: Honestly, work is so hard right now."},
                {"type": "text", "text": "Taylor: Same here. I barely sleep."},
                {"type": "text", "text": "Jordan: I’m so overwhelmed I feel like I could kill myself."},
                {"type": "text", "text": "Alex: Please don’t say that. Let’s take a break and talk if you need to."}
            ]
        ]

        results = []
        for i, convo in enumerate(conversations, start=1):
            print(f"\n=== Running Moderation demo conversation {i} ===")
            messages = self._messages_from_conversation(convo, include_all=False)
            demo_payload = dict(payload)
            demo_payload["messages"] = messages
            res = self.call_agent(demo_payload)
            print(res)
            results.append(res)
        return results[-1] if results else {"result": {"event": []}}