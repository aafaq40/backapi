import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load .env vars
load_dotenv()


# Init Groq client
client = Groq()

# Init app
app = FastAPI()

# Allow local/frontend calls if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema for POST request
class ChatRequest(BaseModel):
    message: str

# System prompt and chat memory
SYSTEM_PROMPT = """
You are a smart conversational booking assistant.
Also you know both english and dutch languages (Don't support any other any language, if user asks for any other languague
just Reply something like, "I support only English and Dutch"), so please lookout at the users preferred language
and do the whole conversation in that language.

You can help users with:
- create_booking
- update_booking
- delete_booking
- cancel

You ask questions naturally and collect missing details step by step.
Do not give the final payload until all required details are provided.

Fields you care about:
- booking_id (for update or delete)
- space_type (desk, room, parking)
- location
- date
- time
- duration
- amenities

If the user says something like "I don't want to book", cancel the booking flow and respond politely.

At the end, respond with final JSON like:
{
  "intent": "create_booking",
  "payload": {
    "location": "...",
    "space_type": "...",
    ...
  }
}
"""

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# Utility to extract structured JSON from LLM reply
def extract_json(text: str):
    try:
        start = text.index("{")
        return json.loads(text[start:])
    except Exception:
        return None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.message
    chat_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=chat_history,
        temperature=0.3
    )

    assistant_msg = response.choices[0].message.content.strip()
    chat_history.append({"role": "assistant", "content": assistant_msg})

    final_payload = extract_json(assistant_msg)

    return {
        "response": assistant_msg,
        "payload": final_payload  # This will be None if not yet ready
    }
