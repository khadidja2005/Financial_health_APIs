import os
import warnings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login
# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

os.environ["USE_TORCH"] = "true"
# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face API token from environment variables
api_token = os.getenv("HUGGING_FACE_TOKEN")
login(api_token)
# Debug: Print the token to ensure it's being retrieved correctly
# print(f"Hugging Face API Token: {api_token}")

# Define the Hugging model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
ConversationModel = HuggingFaceEndpoint(
    repo_id=model_name,
    model_kwargs={
        "huggingface_api_token": api_token,
        "model_type": "text-generation"
    }
)

# Step 1: Load Text Data into Documents
def load_text_file(file_path):
    # Get the directory of the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path
    abs_file_path = os.path.abspath(os.path.join(base_dir, file_path))
    
    with open(abs_file_path, "r", encoding="utf-8") as file:
        content = file.read()
    # Split the content into smaller chunks for better retrieval
    paragraphs = content.split("\n\n")
    # Use `Document` class from `langchain.schema`
    documents = [Document(page_content=para.strip(), metadata={"source": f"Paragraph {i+1}"})
                 for i, para in enumerate(paragraphs) if para.strip()]
    return documents

docs = load_text_file("../../data/platform_info.txt")

# Step 2: Create Embeddings and Build FAISS Index
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

vector_db = FAISS.from_documents(docs, embeddings)

# Step 3: Define the Prompt Template for RAG
rag_prompt_template = PromptTemplate(
    template="The user asked: {question}\nBased on the following context, respond appropriately.\nContext: {context}\nAnswer:",
    input_variables=["question", "context"]
)

def process_user_inputRag(input_text):
    # Retrieve the top 2 relevant documents
    relevant_docs = vector_db.similarity_search(input_text, k=2)
    
    # Combine the content of retrieved documents as context
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    # Create the RAG prompt
    rag_prompt = rag_prompt_template.format(question=input_text, context=context)
    
    # Generate a response using the LLM
    response = ConversationModel(rag_prompt)
    return response