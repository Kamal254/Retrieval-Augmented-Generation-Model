import os
import torch
from pypdf import PdfReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.node_parser import SentenceSplitter

# System Prompt
SYSTEM_PROMPT = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""

QUERY_PROMPT = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

# Load Documents
def load_documents(data_path="Data"):
    documents = []
    if os.path.isdir(data_path):
        documents = SimpleDirectoryReader(data_path).load_data()
    elif data_path.endswith(".pdf"):
        documents = load_pdf_data(data_path)
    return documents

# Load PDF Data
def load_pdf_data(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return [text]

# Initialize Language Model
def initialize_llm():
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=SYSTEM_PROMPT,
        query_wrapper_prompt=QUERY_PROMPT,
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
    )
    return llm

# Initialize Embedding Model
def initialize_embedding_model():
    embedding_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )
    return embedding_model

# Build VectorStore Index
def build_index(documents, llm, embedding_model):
    # Configure settings for the LLM and embedding model
    Settings.llm = llm
    Settings.embed_model = embedding_model
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900

    # Create and return the index
    index = VectorStoreIndex.from_documents(documents, service_context=Settings)
    return index

# Query the Documents
def query_documents(index, query_text):
    query_engine = index.as_query_engine()
    response = query_engine.query(query_text)
    return response

# Main Execution
if __name__ == "__main__":
    # Load documents from the data folder or a specific PDF
    data_path = "Data"  # Update this path to your PDF file if needed
    documents = load_documents(data_path)
    
    # Initialize the models
    llm = initialize_llm()
    embedding_model = initialize_embedding_model()
    
    # Build the index
    index = build_index(documents, llm, embedding_model)
    
    # Query the index
    query_text = "What happened in Sri Venkateswara Swamy Temple in Tirumala and what does Kalyan say?"
    response = query_documents(index, query_text)
    
    print(response)
