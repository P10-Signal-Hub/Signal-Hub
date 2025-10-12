# API test for moduration.py
from fastapi import FastAPI
from pydantic import BaseModel
from Moderation import handle_incoming_message, init_moderator

app = FastAPI(title="Moderation API")

@app.on_event("startup")
async def startup_event():
    # Initialize model once on server startup
    init_moderator()

class MessageRequest(BaseModel):
    username: str
    message: str

@app.post("/moderate")
async def moderate(request: MessageRequest):
    """
    Accepts JSON:
    {
        "username": "alice",
        "message": "Hello world!"
    }
    """
    return handle_incoming_message(request.username, request.message)
