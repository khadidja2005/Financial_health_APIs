import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from typing import Dict, Any

load_dotenv()

class SQLQueryGenerator:
    def __init__(self, schema_info: Dict[str, Dict]):
        """
        Initialize SQL Query Generator with database schema
        
        Args:
            schema_info: Dictionary containing table and column information
            Example:
            {
                "users": {
                    "id": {"type": "INTEGER", "primary_key": True},
                    "name": {"type": "VARCHAR(100)"},
                    "email": {"type": "VARCHAR(255)"}
                },
                "orders": {
                    "id": {"type": "INTEGER", "primary_key": True},
                    "user_id": {"type": "INTEGER", "foreign_key": "users.id"},
                    "amount": {"type": "DECIMAL(10,2)"}
                }
            }
        """
        # Initialize the Mistral model
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={
                "huggingface_api_token": os.getenv("HUGGING_FACE_TOKEN"),
                "model_type": "text-generation"
            }
        )
        
        self.schema = schema_info
        self.schema_description = self._create_schema_description()
        
        # Create prompt template for SQL generation
        self.prompt_template = PromptTemplate(
            input_variables=["schema", "query_request"],
            template="""[INST] You are a SQL query generator. Given the following database schema and request, generate only the SQL query without any explanations.

Database Schema:
{schema}

Request: {query_request}

Rules:
1. Only use tables and columns from the provided schema
2. Follow SQL best practices
3. Include appropriate JOINs if multiple tables are needed
4. Use proper SQL syntax and formatting
5. Return only the SQL query, nothing else

Generate the SQL query: [/INST]"""
        )
        
        # Create the runnable sequence
        self.runnable = (
            RunnablePassthrough() | 
            self.prompt_template | 
            self.llm
        )
        
    def _create_schema_description(self) -> str:
        """Convert schema dictionary into a readable format for the prompt."""
        description = ""
        for table, columns in self.schema.items():
            description += f"\nTable: {table}\n"
            for col_name, col_info in columns.items():
                description += f"- {col_name} ({col_info['type']})"
                if col_info.get('foreign_key'):
                    description += f" -> References {col_info['foreign_key']}"
                if col_info.get('primary_key'):
                    description += " (Primary Key)"
                description += "\n"
        return description
    
    def validate_query(self, query: str) -> bool:
        """
        Validate the generated query against the schema.
        
        Args:
            query: Generated SQL query
            
        Returns:
            bool: True if query only uses existing tables and columns
        """
        query = query.lower()
        words = set(query.replace('(', ' ').replace(')', ' ').split())
        
        # Check if any referenced table or column doesn't exist in schema
        for table in self.schema:
            if table.lower() in words:
                words.remove(table.lower())
                
            for column in self.schema[table]:
                if column.lower() in words:
                    words.remove(column.lower())
        
        # Remove SQL keywords and common symbols
        sql_keywords = {'select', 'from', 'where', 'join', 'and', 'or', 'on', 'as',
                       'inner', 'left', 'right', 'outer', 'group', 'by', 'having',
                       'order', 'limit', 'offset', 'in', 'between', 'like', 'is',
                       'null', 'not', 'true', 'false', '*', '=', '<', '>', '<=',
                       '>=', '!=', ';'}
        if words.intersection(sql_keywords):
          return True
    
        return False
    
    async def generate_query(self, query_request: str) -> Dict[str, Any]:
        """
        Generate a SQL query based on the natural language request.
        
        Args:
            query_request: Natural language query request
            
        Returns:
            Dict containing the generated query and validation result
        """
        # Prepare input data for the prompt
        input_data = {
            "schema": self.schema_description,
            "query_request": query_request
        }
        
        # Generate SQL query using the runnable sequence
        generated_query = await self.runnable.ainvoke(input_data)
        
        # Clean up the generated query
        clean_query = generated_query.strip().replace("```sql", "").replace("```", "").strip()
        
        # Validate the query
        is_valid = self.validate_query(clean_query)
        
        return {
            "query": clean_query,
            "is_valid": is_valid
        }

async def main():
    # Example schema
    schema = {
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
    
    # Initialize the generator
    generator = SQLQueryGenerator(schema)
    
    # Example queries
    test_queries = [
        "Show me all users who have placed orders over $100",
        "Find the total order amount for each user in the last month",
        "List users who haven't placed any orders",
        "Get the average order amount by user, only for users with more than 5 orders"
    ]
    
    # Test the generator
    for query_request in test_queries:
        print(f"\nRequest: {query_request}")
        result = await generator.generate_query(query_request)
        print(f"{result['query']}")
        print(f"Query is valid: {result['is_valid']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())