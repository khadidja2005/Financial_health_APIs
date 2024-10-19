from fastapi import FastAPI, HTTPException , APIRouter
from pydantic import BaseModel
from typing import List
import numpy as np
from .Recommander import ExpenseAdvisor  

routeRecommander = APIRouter()

# Request model
class ExpenseRequest(BaseModel):
    expenses: List[float]

# Response model
class ExpenseRecommendation(BaseModel):
    recommendations: str
    metrics: dict

# Initialize ExpenseAdvisor
advisor = ExpenseAdvisor()

@routeRecommander.post("/analyze-expenses", response_model=ExpenseRecommendation)
async def analyze_expenses(request: ExpenseRequest):
    """
    Analyze expenses and generate recommendations
    
    Args:
        request: ExpenseRequest containing an array of daily expenses
        
    Returns:
        ExpenseRecommendation containing recommendations and metrics
    """
    try:
        # Validate input length
        if len(request.expenses) < 7:
            raise HTTPException(
                status_code=400,
                detail="Please provide at least 7 days of expense data"
            )
            
        if len(request.expenses) > 90:
            raise HTTPException(
                status_code=400,
                detail="Maximum 90 days of expense data allowed"
            )

        # Convert expenses to numpy array
        expenses_array = np.array(request.expenses)
        
        # Validate expense values
        if np.any(expenses_array < 0):
            raise HTTPException(
                status_code=400,
                detail="Expenses cannot be negative"
            )

        # Get metrics
        metrics_dict = advisor.preprocess_expenses(expenses_array)
        
        # Generate recommendations
        recommendations = await advisor.generate_recommendations(expenses_array)
        
        return ExpenseRecommendation(
            recommendations=recommendations,
            metrics=metrics_dict
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
