import uuid
from typing import Dict, Any

from fastapi import FastAPI

from Model import Model
import EventAgent
import ModerationAgent
import SummarizerAgent

app = FastAPI()

class AgentHandler:
    def __init__(self):
        self.model = Model()
    #Event creation
    def call_event_agent(self, payload):
        agent = EventAgent.EventAgent(self.model)
        #Test agent
        if 'test' in payload and payload['test'] == True:
            return agent.test_agent(payload)
        else:
            return agent.call_agent(payload)
    #Moderation creation
    def call_moderation_agent(self, payload):
        agent = ModerationAgent.ModerationAgent(self.model) 
        if 'test' in payload and payload['test'] == True:
            return agent.test_agent(payload)
        else:
            return agent.call_agent(payload)
    #Summarizer creation
    def call_summarizer_agent(self, payload):          # <-- add this
        agent = SummarizerAgent.SummarizerAgent(self.model)
        if 'test' in payload and payload['test'] == True:
            return agent.test_agent(payload)
        else:
            return agent.call_agent(payload)

handler = AgentHandler()

@app.post("/agent/use")
def handle_event(request: Dict[str, Any]):
    payload = request.get("payload")
    method = request.get("method")
    payload['request_id'] = uuid.uuid4()
    if method == "event_creation":
        return handler.call_event_agent(payload)
    elif method == "moderation":
        return handler.call_moderation_agent(payload)
    elif method == "summarize":
        return handler.call_summarizer_agent(payload)
    return None

test_event = {
    'method': 'event_creation',
    "payload": {
        "test": True,
    }
}
handle_event(test_event)

test_moderation = {
    'method': 'moderation',
    'payload': {
        'test': True
    }
}
handle_event(test_moderation)