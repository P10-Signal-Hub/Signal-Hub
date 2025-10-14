import uuid
from typing import Dict, Any

from fastapi import FastAPI

from Model import Model
import EventAgent

app = FastAPI()

class AgentHandler:
    def __init__(self):
        self.model = Model()

    @app.post("/agent/use")
    def handle_event(self, request: Dict[str, Any]):
        payload = request.get("payload")
        method = request.get("method")
        payload['request_id'] = uuid.uuid4()
        if method == "event_creation":
            agent = EventAgent.EventAgent(self.model)
            if 'test' in payload and payload['test'] == True:
                return agent.test_agent(payload)
            else:
                return agent.input_handler(payload)
        return None

handler = AgentHandler()
test_event = {
    'method': 'event_creation',
    "payload": {
        "test": True,
    }
}
handler.handle_event(test_event)
