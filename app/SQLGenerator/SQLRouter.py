# router.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from .Model import SQLQueryGenerator as SQLGenerator

load_dotenv()

routerSQL = APIRouter()

class QueryRequest(BaseModel):
    prompt: str
    execute: Optional[bool] = False

class QueryResponse(BaseModel):
    query: str
    is_valid: bool
    results: Optional[list] = None

# Example schema - replace with your actual schema
SCHEMA = {
    "users": {
        "id": {"type": "INTEGER", "primary_key": True},
        "name": {"type": "VARCHAR(100)"},
        "email": {"type": "VARCHAR(255)"},
        "created_at": {"type": "TIMESTAMP"}
    },
    "orders": {
        "id": {"type": "INTEGER", "primary_key": True},
        "user_id": {"type": "INTEGER", "foreign_key": "users.id"},
        "total_amount": {"type": "DECIMAL(10,2)"},
        "status": {"type": "VARCHAR(50)"},
        "order_date": {"type": "TIMESTAMP"}
    }
}

# Create a generator instance
sql_generator = SQLGenerator(SCHEMA)

# Dependency to get the generator
async def get_generator():
    return sql_generator

@routerSQL.post("/generate", response_model=QueryResponse)
async def generate_sql_query(
    request: QueryRequest,
    generator: SQLGenerator = Depends(get_generator)
):
    """Generate SQL query from natural language prompt"""
    try:
        # Generate the query
        result = await generator.generate_query(request.prompt)
        
        response = QueryResponse(
            query=result["query"],
            is_valid=result["is_valid"],
            results=None  # Add query execution logic here if needed
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))