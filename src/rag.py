import os

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

# --- Embedding (BGE) ---
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),  # uses OPENAI_API_KEY env var by default
)

# --- Apply to Global Settings ---
Settings.embed_model = embed_model
Settings.llm = llm
Settings.context_window = 2048  # or slightly lower, like 2000 for safety
Settings.chunk_size = 512       # to control size of document chunks
Settings.chunk_overlap = 50 

# --- Load Storage for each FAISS Index ---
storage_hr = StorageContext.from_defaults(persist_dir="./storage_faiss_hr_docs")
storage_strategy = StorageContext.from_defaults(persist_dir="./storage_faiss_strategy_docs")

index_hr = load_index_from_storage(storage_hr)
index_strategy = load_index_from_storage(storage_strategy)

# Create query engines from them
query_engine_hr = index_hr.as_query_engine(similarity_top_k=5)
query_engine_strategy = index_strategy.as_query_engine(similarity_top_k=5)
def query_hr(input):
    return query_engine_hr.query(input)
def query_strategy(input):
    return query_engine_strategy.query(input)