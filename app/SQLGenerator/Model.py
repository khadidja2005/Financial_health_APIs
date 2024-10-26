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
    Exemple :{
    "Organization": {
        "id": {"type": "VARCHAR", "primary_key": True, "default": "cuid()"},
        "name": {"type": "VARCHAR"},
        "ownerId": {"type": "VARCHAR", "foreign_key": "User.id"},
        "createdAt": {"type": "TIMESTAMP", "default": "now()"},
        "updatedAt": {"type": "TIMESTAMP", "default": "updatedAt"}
    },
    "CashFlow": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "date": {"type": "TIMESTAMP"},
        "cash_inflow": {"type": "DECIMAL"},
        "cash_outflow": {"type": "DECIMAL"},
        "net_cash_flow": {"type": "DECIMAL"},
        "description": {"type": "TEXT", "nullable": True},
        "category": {"type": "VARCHAR(50)", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Expenses": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "date": {"type": "TIMESTAMP"},
        "amount": {"type": "DECIMAL"},
        "expense_category": {"type": "VARCHAR(50)", "nullable": True},
        "department": {"type": "VARCHAR(50)", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Revenue": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "date": {"type": "TIMESTAMP"},
        "amount": {"type": "DECIMAL"},
        "product_line": {"type": "VARCHAR(50)", "nullable": True},
        "customer_type": {"type": "VARCHAR(50)", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Profit": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "date": {"type": "TIMESTAMP"},
        "revenue": {"type": "DECIMAL"},
        "expenses": {"type": "DECIMAL"},
        "net_profit": {"type": "DECIMAL"},
        "profit_margin": {"type": "DECIMAL", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Budget": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "fiscal_year": {"type": "VARCHAR(50)", "nullable": True},
        "department": {"type": "VARCHAR(50)", "nullable": True},
        "allocated_budget": {"type": "DECIMAL"},
        "spent_budget": {"type": "DECIMAL"},
        "remaining_budget": {"type": "DECIMAL"},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Debt": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "debt_type": {"type": "VARCHAR(50)", "nullable": True},
        "principal": {"type": "DECIMAL"},
        "interest_rate": {"type": "DECIMAL", "nullable": True},
        "maturity_date": {"type": "TIMESTAMP", "nullable": True},
        "payment_due_date": {"type": "TIMESTAMP", "nullable": True},
        "amount_paid": {"type": "DECIMAL", "nullable": True},
        "outstanding_balance": {"type": "DECIMAL", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Investments": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "investment_type": {"type": "VARCHAR(50)", "nullable": True},
        "investment_amount": {"type": "DECIMAL"},
        "investment_date": {"type": "TIMESTAMP"},
        "returns": {"type": "DECIMAL", "nullable": True},
        "risk_level": {"type": "VARCHAR(50)", "nullable": True},
        "current_value": {"type": "DECIMAL", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Funding": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "funding_round": {"type": "VARCHAR(50)", "nullable": True},
        "amount_raised": {"type": "DECIMAL"},
        "date": {"type": "TIMESTAMP"},
        "investor_name": {"type": "VARCHAR(50)", "nullable": True},
        "valuation": {"type": "DECIMAL", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "FinancialReports": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "report_type": {"type": "VARCHAR(50)", "nullable": True},
        "start_date": {"type": "TIMESTAMP"},
        "end_date": {"type": "TIMESTAMP"},
        "content": {"type": "TEXT", "nullable": True},
        "created_at": {"type": "TIMESTAMP", "default": "now()"},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
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
    schema ={
    "Organization": {
        "id": {"type": "VARCHAR", "primary_key": True, "default": "cuid()"},
        "name": {"type": "VARCHAR"},
        "ownerId": {"type": "VARCHAR", "foreign_key": "User.id"},
        "createdAt": {"type": "TIMESTAMP", "default": "now()"},
        "updatedAt": {"type": "TIMESTAMP", "default": "updatedAt"}
    },
    "CashFlow": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "date": {"type": "TIMESTAMP"},
        "cash_inflow": {"type": "DECIMAL"},
        "cash_outflow": {"type": "DECIMAL"},
        "net_cash_flow": {"type": "DECIMAL"},
        "description": {"type": "TEXT", "nullable": True},
        "category": {"type": "VARCHAR(50)", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Expenses": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "date": {"type": "TIMESTAMP"},
        "amount": {"type": "DECIMAL"},
        "expense_category": {"type": "VARCHAR(50)", "nullable": True},
        "department": {"type": "VARCHAR(50)", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Revenue": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "date": {"type": "TIMESTAMP"},
        "amount": {"type": "DECIMAL"},
        "product_line": {"type": "VARCHAR(50)", "nullable": True},
        "customer_type": {"type": "VARCHAR(50)", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Profit": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "date": {"type": "TIMESTAMP"},
        "revenue": {"type": "DECIMAL"},
        "expenses": {"type": "DECIMAL"},
        "net_profit": {"type": "DECIMAL"},
        "profit_margin": {"type": "DECIMAL", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Budget": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "fiscal_year": {"type": "VARCHAR(50)", "nullable": True},
        "department": {"type": "VARCHAR(50)", "nullable": True},
        "allocated_budget": {"type": "DECIMAL"},
        "spent_budget": {"type": "DECIMAL"},
        "remaining_budget": {"type": "DECIMAL"},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Debt": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "debt_type": {"type": "VARCHAR(50)", "nullable": True},
        "principal": {"type": "DECIMAL"},
        "interest_rate": {"type": "DECIMAL", "nullable": True},
        "maturity_date": {"type": "TIMESTAMP", "nullable": True},
        "payment_due_date": {"type": "TIMESTAMP", "nullable": True},
        "amount_paid": {"type": "DECIMAL", "nullable": True},
        "outstanding_balance": {"type": "DECIMAL", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Investments": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "investment_type": {"type": "VARCHAR(50)", "nullable": True},
        "investment_amount": {"type": "DECIMAL"},
        "investment_date": {"type": "TIMESTAMP"},
        "returns": {"type": "DECIMAL", "nullable": True},
        "risk_level": {"type": "VARCHAR(50)", "nullable": True},
        "current_value": {"type": "DECIMAL", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "Funding": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "funding_round": {"type": "VARCHAR(50)", "nullable": True},
        "amount_raised": {"type": "DECIMAL"},
        "date": {"type": "TIMESTAMP"},
        "investor_name": {"type": "VARCHAR(50)", "nullable": True},
        "valuation": {"type": "DECIMAL", "nullable": True},
        "description": {"type": "TEXT", "nullable": True},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    },
    "FinancialReports": {
        "id": {"type": "INTEGER", "primary_key": True, "auto_increment": True},
        "report_type": {"type": "VARCHAR(50)", "nullable": True},
        "start_date": {"type": "TIMESTAMP"},
        "end_date": {"type": "TIMESTAMP"},
        "content": {"type": "TEXT", "nullable": True},
        "created_at": {"type": "TIMESTAMP", "default": "now()"},
        "organizationId": {"type": "VARCHAR", "foreign_key": "Organization.id"}
    }
}

    # Initialize the generator
    generator = SQLQueryGenerator(schema)
    
    # Example queries
    test_queries = [
    "List all debts with outstanding balances over $10,000 and their maturity dates",
    "Get the average expense amount by department, only for departments with more than 5 expenses",
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