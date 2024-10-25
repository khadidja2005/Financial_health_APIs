from fastapi import FastAPI
import uvicorn
from app.controllers.chatbot.Chatbot_controller import  router as chatbot_router
from app.services.ExpenseModels.router import routerEX as expenses_router
from app.services.ExpenseModels.Recom_router import routeRecommander as recommander_router
from app.services.GoalPlanner.GoalRouter import routeGoal as goal_router
from app.SQLGenerator.SQLRouter import routerSQL as sql_router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router( expenses_router , prefix="/expenses")
app.include_router(chatbot_router, prefix="/chatbot")
app.include_router(recommander_router, prefix="/recommand")
app.include_router(goal_router, prefix="/goal")
app.include_router(sql_router , prefix="/sql")
@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

if __name__ == "__main__":
    # Specify the port here within the code
    uvicorn.run(app, host="0.0.0.0", port=8080)
