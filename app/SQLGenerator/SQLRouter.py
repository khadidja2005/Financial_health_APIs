from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from .Model import SQLQueryGenerator as SQLGenerator
from .Modeltrained import RAGSQLQueryGenerator 
load_dotenv()

routerSQL = APIRouter()

# Pydantic models
class QueryRequest(BaseModel):
    query: str  # Changed from prompt to query to match the generator's interface
    execute: Optional[bool] = False

class QueryResponse(BaseModel):
    query: str
    is_valid: bool
    #results: Optional[List[Any]] = None

class QueryResponseRAG(BaseModel):
    query: str
    is_valid: bool
    #similar_examples: str

# Schema and example pairs
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

EXAMPLE_PAIRS = [
    (
        "Show me all users who have placed orders over $100",
        """SELECT DISTINCT u.* 
           FROM users u
           JOIN orders o ON u.id = o.user_id
           WHERE o.total_amount > 100;"""
    ),
    (
        "Find the total order amount for each user in the last month",
        """SELECT u.name, SUM(o.total_amount) as total_spent
           FROM users u
           JOIN orders o ON u.id = o.user_id
           WHERE o.order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
           GROUP BY u.id, u.name;"""
    )
]

# Create generator instances
sql_generator = SQLGenerator(SCHEMA)
rag_sql_generator = RAGSQLQueryGenerator(SCHEMA, EXAMPLE_PAIRS)

# Dependencies
async def get_generator():
    return sql_generator

async def get_rag_generator():
    return rag_sql_generator

@routerSQL.post("/generate", response_model=QueryResponse)
async def generate_sql_query(
    request: QueryRequest,
    generator: SQLGenerator = Depends(get_generator)
):
    """Generate SQL query from natural language prompt"""
    try:
        # Generate the query
        result = await generator.generate_query(request.query)  # Changed from prompt to query
        
        return QueryResponse(
            query=result["query"],
            is_valid=result["is_valid"],
            #results=None
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate SQL query: {str(e)}"
        )

@routerSQL.post("/generate-rag", response_model=QueryResponseRAG)
async def generate_rag_sql_query(
    request: QueryRequest,
    generator: RAGSQLQueryGenerator = Depends(get_rag_generator)
):
    """Generate SQL query using RAG model from natural language prompt"""
    try:
        # Generate the query
        result = await generator.generate_query(request.query)  # Changed from prompt to query
        
        return QueryResponseRAG(
            query=result["query"],
            is_valid=result["is_valid"],
            #similar_examples=result["similar_examples"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate SQL query with RAG: {str(e)}"
        )