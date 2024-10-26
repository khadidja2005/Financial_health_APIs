# chatbot_service/chatbot_controller.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.InstructionGuide.Model import process_user_inputRag

# Define a router for the chatbot
router = APIRouter()

# Define the request body schema using Pydantic
class ChatbotRequest(BaseModel):
    input_text: str

@router.post("/assistant")
async def ask_question(input_text: ChatbotRequest):
    try:
        response = process_user_inputRag(input_text.input_text)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
