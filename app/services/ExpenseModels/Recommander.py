import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from typing import Dict, Any

load_dotenv()

class ExpenseAdvisor:
    def __init__(self):
        # Initialize the model
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                model_kwargs={
        "huggingface_api_token": os.getenv("HUGGING_FACE_TOKEN"),
        "model_type": "text-generation" , 
    } 
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["metrics"],
            template="""You are a financial advisor analyzing a company's expenses over the last 30 periods.

{metrics}

Based on this expense data, provide:
1. Recommendations for future expense planning 
2. Budget optimization suggestions
Format your response in clear sections with bullet points. Focus on actionable insights and quantifiable targets."""
        )
        
        # Create the runnable sequence
        self.runnable = (
            RunnablePassthrough() | 
            self.prompt_template | 
            self.llm
        )
        
    def preprocess_expenses(self, expenses: np.ndarray) -> Dict[str, Any]:
        """
        Prepare expense metrics for the prompt
        """
        df = pd.DataFrame({'daily_expenses': expenses})
        
        # Calculate key metrics
        weekday_avg = df.groupby(df.index % 7)['daily_expenses'].mean()
        metrics = {
            'average_expense': df['daily_expenses'].mean(),
            'expense_std': df['daily_expenses'].std(),
            'highest_expense': df['daily_expenses'].max(),
            'lowest_expense': df['daily_expenses'].min(),
            'highest_spending_day': df['daily_expenses'].idxmax() + 1,  
            'weekly_pattern': weekday_avg.std() / weekday_avg.mean() > 0.1,  
            'expense_trend': np.polyfit(np.arange(len(df)), df['daily_expenses'], 1)[0],  
            'spending_spikes': len(df[df['daily_expenses'] > df['daily_expenses'].mean() + 2*df['daily_expenses'].std()]),
            'consistent_days': len(df[abs(df['daily_expenses'] - df['daily_expenses'].mean()) < df['daily_expenses'].std()]),
        }
        
        # Format metrics for the prompt
        metrics_text = f"""Daily Expense Analysis:
- Average Daily Expense: ${metrics['average_expense']:,.2f}
- Day-to-Day Volatility: ${metrics['expense_std']:,.2f}
- Highest Daily Expense: ${metrics['highest_expense']:,.2f} (Day {metrics['highest_spending_day']})
- Lowest Daily Expense: ${metrics['lowest_expense']:,.2f}
- Daily Trend: ${metrics['expense_trend']:,.2f} per day
- Number of Spending Spikes: {metrics['spending_spikes']} days
- Days with Normal Spending: {metrics['consistent_days']} days
- Weekly Pattern Detected: {'Yes' if metrics['weekly_pattern'] else 'No'}

Weekly Breakdown:
{pd.DataFrame({'Average Expense': weekday_avg}).to_string()}"""
        
        return {"metrics": metrics_text}
    
    async def generate_recommendations(self, expenses: np.ndarray) -> str:
        """
        Generate expense recommendations based on historical data
        """
        # Preprocess the expense data
        input_data = self.preprocess_expenses(expenses)
        
        # Generate recommendations using the runnable sequence
        recommendations = await self.runnable.ainvoke(input_data)
        
        return recommendations

async def main():
    # Sample data for demonstration
    expenses = np.random.normal(100000, 10000, 30)  # 30 periods of expenses
    advisor = ExpenseAdvisor()
    recommendations = await advisor.generate_recommendations(expenses)
    print("\nGenerated Recommendations:\n")
    print(recommendations)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())