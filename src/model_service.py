import os
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
model_type = "text-generation"

env = os.getenv("APP_ENV")
if env == "prod":
    print("Running in production mode.")
    # LLM_MODEL = "Qwen/Qwen3-8B"
    LLM_MODEL = "FutureMa/Eva-4B"
else:
    print("Running in development mode.")
    LLM_MODEL = "Qwen/Qwen3-0.6B"

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    dtype="auto",
    trust_remote_code=True
)

qa_pipeline = pipeline(
    model_type,
    model=llm_model,
    tokenizer=tokenizer,
    return_full_text=False   # 只要答案部分
)

def get_embedder():
    return embed_fn

def get_pipeline():
    return qa_pipeline