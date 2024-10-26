import os
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import asyncio
import nest_asyncio
nest_asyncio.apply()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
load_dotenv()

class RAGSQLQueryGenerator:
    def __init__(self, schema_info: Dict[str, Dict], example_pairs: List[Tuple[str, str]] = None):
        """
        Initialize RAG-enhanced SQL Query Generator
        
        Args:
            schema_info: Dictionary containing table and column information
            example_pairs: List of (query_request, sql_query) tuples for training
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
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store from example pairs
        self.vector_store = self._create_vector_store(example_pairs or [])
        
        # Enhanced prompt template with examples and RAG
        self.prompt_template = PromptTemplate(
            input_variables=["schema", "query_request", "similar_examples"],
            template="""[INST] You are a SQL query generator specialized in working with our specific database schema. Given the schema, request, and similar examples, generate a SQL query.

Database Schema:
{schema}

Similar Examples:
{similar_examples}

Request: {query_request}

Rules:
1. Only use tables and columns from the provided schema
2. Follow SQL best practices and patterns shown in the examples
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
    
    def _create_vector_store(self, example_pairs: List[Tuple[str, str]]) -> FAISS:
        """Create FAISS vector store from example query pairs."""
        documents = []
        for request, query in example_pairs:
            doc_content = f"Request: {request}\nQuery: {query}"
            documents.append(Document(page_content=doc_content))
        
        if documents:
            return FAISS.from_documents(documents, self.embeddings)
        else:
            # Create empty vector store if no examples
            return FAISS.from_documents([Document(page_content="Empty initial store")], self.embeddings)
    
    def add_example(self, request: str, query: str):
        """
        Add a new example query pair to the vector store.
        
        Args:
            request: Natural language query request
            query: Corresponding SQL query
        """
        doc_content = f"Request: {request}\nQuery: {query}"
        self.vector_store.add_documents([Document(page_content=doc_content)])
    
    def get_similar_examples(self, query_request: str, k: int = 2) -> str:
        """
        Retrieve similar example queries from the vector store.
        
        Args:
            query_request: Natural language query request
            k: Number of similar examples to retrieve
            
        Returns:
            Formatted string of similar examples
        """
        similar_docs = self.vector_store.similarity_search(query_request, k=k)
        return "\n\n".join(doc.page_content for doc in similar_docs)
    
    def validate_query(self, query: str) -> bool:
        """Validate the generated query against the schema."""
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
        Generate a SQL query based on the natural language request and schema .
        
        Args:
            query_request: Natural language query request
            
        Returns:
            Dict containing the generated query and validation result
        """
        # Get similar examples
        similar_examples = self.get_similar_examples(query_request)
        
        # Prepare input data for the prompt
        input_data = {
            "schema": self.schema_description,
            "query_request": query_request,
            "similar_examples": similar_examples
        }
        
        # Generate SQL query using the runnable sequence
        generated_query = await self.runnable.ainvoke(input_data)
        
        # Clean up the generated query
        clean_query = generated_query.strip().replace("```sql", "").replace("```", "").strip()
        
        # Validate the query
        is_valid = self.validate_query(clean_query)
        
        return {
            "query": clean_query,
            "is_valid": is_valid,
            "similar_examples": similar_examples
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

    
    # Example query pairs for training
    example_pairs = [
    (
        "List all organizations with their total cash inflow and outflow",
        """SELECT o.name, 
                  SUM(c.cash_inflow) AS total_cash_inflow,
                  SUM(c.cash_outflow) AS total_cash_outflow
           FROM Organization o
           JOIN CashFlow c ON o.id = c.organizationId
           GROUP BY o.name;"""
    ),
    (
        "Retrieve all expenses in the 'Marketing' department for the current year",
        """SELECT e.date, e.amount, e.expense_category, e.description
           FROM Expenses e
           JOIN Organization o ON e.organizationId = o.id
           WHERE e.department = 'Marketing'
             AND EXTRACT(YEAR FROM e.date) = EXTRACT(YEAR FROM CURRENT_DATE);"""
    ),
    (
        "Find all revenue entries with a 'Corporate' customer type",
        """SELECT r.date, r.amount, r.product_line, r.description
           FROM Revenue r
           JOIN Organization o ON r.organizationId = o.id
           WHERE r.customer_type = 'Corporate';"""
    ),
    (
        "Calculate the net profit and profit margin for each organization",
        """SELECT o.name,
                  SUM(p.net_profit) AS total_net_profit,
                  AVG(p.profit_margin) AS average_profit_margin
           FROM Organization o
           JOIN Profit p ON o.id = p.organizationId
           GROUP BY o.name;"""
    ),
    (
        "Find all investments with a 'High' risk level and their current value",
        """SELECT i.investment_type, i.investment_amount, i.current_value, i.description
           FROM Investments i
           JOIN Organization o ON i.organizationId = o.id
           WHERE i.risk_level = 'High';"""
    ),
    (
        "Show the total amount raised in each funding round for each organization",
        """SELECT o.name, f.funding_round, SUM(f.amount_raised) AS total_amount_raised
           FROM Organization o
           JOIN Funding f ON o.id = f.organizationId
           GROUP BY o.name, f.funding_round;"""
    ),
    (
        "List all debts with outstanding balances over $10,000 and their maturity dates",
        """SELECT d.debt_type, d.principal, d.outstanding_balance, d.maturity_date
           FROM Debt d
           JOIN Organization o ON d.organizationId = o.id
           WHERE d.outstanding_balance > 10000;"""
    ),
    (
        "Get budget details for each department with remaining budgets",
        """SELECT b.department, b.allocated_budget, b.spent_budget, b.remaining_budget
           FROM Budget b
           JOIN Organization o ON b.organizationId = o.id
           WHERE b.remaining_budget > 0;"""
    ),
    (
        "Generate a list of all financial reports for the last fiscal year",
        """SELECT fr.report_type, fr.start_date, fr.end_date, fr.content
           FROM FinancialReports fr
           JOIN Organization o ON fr.organizationId = o.id
           WHERE fr.start_date >= DATE_TRUNC('year', CURRENT_DATE - INTERVAL '1 year');"""
    ),
    (
        "Calculate net cash flow per organization and categorize by inflow vs. outflow",
        """SELECT o.name, 
                  SUM(CASE WHEN c.cash_inflow > 0 THEN c.cash_inflow ELSE 0 END) AS total_inflow,
                  SUM(CASE WHEN c.cash_outflow > 0 THEN c.cash_outflow ELSE 0 END) AS total_outflow,
                  SUM(c.net_cash_flow) AS net_cash_flow
           FROM Organization o
           JOIN CashFlow c ON o.id = c.organizationId
           GROUP BY o.name;"""
    )
]

    # Initialize the generator with examples
    generator = RAGSQLQueryGenerator(schema, example_pairs)
    
    # Test queries
    test_queries = [
    "Get the average expense amount by department, only for departments with more than 5 expenses",
]
    
    for query_request in test_queries:
        print(f"\nRequest: {query_request}")
        result = await generator.generate_query(query_request)
        #print(f"\nSimilar examples used:\n{result['similar_examples']}")
        print(f"\nGenerated Query:\n{result['query']}")
        print(f"Query is valid: {result['is_valid']}") 
    print("\n=== Query Generation Test Completed ===\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())