from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pandas as pd
from .Recommander import ExpenseAdvisor

routeRecommander = APIRouter() 

# Request model
class ExpenseRequest(BaseModel):
    expenses: List[float]

# Structured metrics models
class WeeklyBreakdown(BaseModel):
    day_0: float
    day_1: float
    day_2: float
    day_3: float
    day_4: float
    day_5: float
    day_6: float

class MetricsResponse(BaseModel):
    average_daily_expense: float
    day_to_day_volatility: float
    highest_daily_expense: float
    lowest_daily_expense: float
    highest_spending_day: int
    daily_trend: float
    spending_spikes: int
    normal_spending_days: int
    weekly_pattern_detected: bool
    weekly_breakdown: WeeklyBreakdown

# Response model
class ExpenseRecommendation(BaseModel):
    recommendations: str
    metrics: MetricsResponse

# Modify the ExpenseAdvisor's preprocess_expenses method
class StructuredExpenseAdvisor(ExpenseAdvisor):
    def preprocess_expenses(self, expenses: np.ndarray) -> Dict:
        """
        Prepare expense metrics as structured data
        """
        df = pd.DataFrame({'daily_expenses': expenses})
        
        # Calculate weekly breakdown
        weekday_avg = df.groupby(df.index % 7)['daily_expenses'].mean()
        weekly_breakdown = {
            f"day_{day}": float(avg) 
            for day, avg in weekday_avg.items()
        }

        # Calculate trend
        trend = np.polyfit(np.arange(len(df)), df['daily_expenses'], 1)[0]
        
        # Structure all metrics
        metrics = MetricsResponse(
            average_daily_expense=float(df['daily_expenses'].mean()),
            day_to_day_volatility=float(df['daily_expenses'].std()),
            highest_daily_expense=float(df['daily_expenses'].max()),
            lowest_daily_expense=float(df['daily_expenses'].min()),
            highest_spending_day=int(df['daily_expenses'].idxmax() + 1),
            daily_trend=float(trend),
            spending_spikes=int(len(df[df['daily_expenses'] > df['daily_expenses'].mean() + 2*df['daily_expenses'].std()])),
            normal_spending_days=int(len(df[abs(df['daily_expenses'] - df['daily_expenses'].mean()) < df['daily_expenses'].std()])),
            weekly_pattern_detected=bool(weekday_avg.std() / weekday_avg.mean() > 0.1),
            weekly_breakdown=WeeklyBreakdown(**weekly_breakdown)
        )
        
        return {"metrics": metrics}

# Initialize advisor
advisor = StructuredExpenseAdvisor()

@routeRecommander.post("/analyze-expenses", response_model=ExpenseRecommendation)
async def analyze_expenses(request: ExpenseRequest):
    """
    Analyze expenses and generate recommendations
    
    Args:
        request: ExpenseRequest containing an array of daily expenses
        
    Returns:
        ExpenseRecommendation containing recommendations and structured metrics
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

        # Get structured metrics
        result = advisor.preprocess_expenses(expenses_array)
        
        # Generate recommendations
        recommendations = await advisor.generate_recommendations(expenses_array)
        
        return ExpenseRecommendation(
            recommendations=recommendations,
            metrics=result["metrics"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )