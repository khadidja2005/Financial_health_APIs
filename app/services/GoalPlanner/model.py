import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from typing import Dict, Any

load_dotenv()

class GoalAdvisor:
    def __init__(self):
        # Initialize the model
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                model_kwargs={
        "huggingface_api_token": os.getenv("HUGGING_FACE_TOKEN"),
        "model_type": "text-generation"
    }
        )
        
        # Create prompt template for goals
        self.prompt_template = PromptTemplate(
            input_variables=["metrics", "goal"],
            template="""You are a financial advisor analyzing a company's expenses over the last 30 periods.
The company has set the following goal:
{goal}

Based on this goal , provide:
1. Steps the company should take to achieve the goal.
2. Recommendations on whether or not the company should pursue this goal (consider financial risk, feasibility, and potential benefits).
Format your response in clear sections with bullet points. Focus on actionable insights and a balanced evaluation of the goal."""
        )
        
        # Create the runnable sequence
        self.runnable = (
            RunnablePassthrough() | 
            self.prompt_template | 
            self.llm
        )
        
    async def generate_recommendations(self, goal: str) -> str:
        input_data = { "goal": goal }
        
        # Generate recommendations using the runnable sequence
        recommendations = await self.runnable.ainvoke(input_data)
        
        return recommendations

async def main():
    
    # User-provided goal
    goal = "we are planning to make a huge investment in the next quarter with international companies"

    advisor = GoalAdvisor()
    recommendations = await advisor.generate_recommendations( goal)
    
    print("\nGenerated Steps and Recommendations:\n")
    print(recommendations)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
