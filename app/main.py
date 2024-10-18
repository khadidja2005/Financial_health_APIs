from fastapi import FastAPI

from app.controllers.chatbot.Chatbot_controller import  router as chatbot_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chatbot_router, prefix="/chatbot")
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

