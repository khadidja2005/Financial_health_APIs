from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import Dict
from .model import GoalAdvisor 

routeGoal = APIRouter()

# Request model for goal-based analysis
class GoalRequest(BaseModel):
    goal: str

# Response model for goal recommendations
class GoalRecommendation(BaseModel):
    recommendations: str

# Initialize the GoalAdvisor
advisor = GoalAdvisor()

@routeGoal.post("/analyze-goal", response_model=GoalRecommendation)
async def analyze_goal(request: GoalRequest):
    """
    Analyze a goal and generate recommendations
    
    Args:
        request: GoalRequest containing the user's goal
    
    Returns:
        GoalRecommendation containing actionable steps and recommendations
    """
    try:
        # Get goal from request
        goal = request.goal
        
        # Ensure that a valid goal is provided
        if not goal:
            raise HTTPException(
                status_code=400,
                detail="A valid goal must be provided"
            )

        # Generate recommendations using the GoalAdvisor
        recommendations = await advisor.generate_recommendations(goal)
        
        return GoalRecommendation(
            recommendations=recommendations
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
