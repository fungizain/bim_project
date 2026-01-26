import os
from chromadb.utils import embedding_functions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GptOssForCausalLM,
    pipeline
)

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

env = os.getenv("APP_ENV")
if env == "prod":
    print("Running in production mode.")
    # LLM_MODEL = "nvidia/Nemotron-Orchestrator-8B"
    LLM_MODEL = "openai/gpt-oss-20b"
else:
    print("Running in development mode.")
    LLM_MODEL = "Qwen/Qwen3-0.6B"

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
autoModel = GptOssForCausalLM if "gpt-oss" in LLM_MODEL else AutoModelForCausalLM
model = autoModel.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    dtype="auto",
    trust_remote_code=True
)

qa_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True
)

def get_embedder():
    return embedder

def get_pipeline():
    return qa_pipeline