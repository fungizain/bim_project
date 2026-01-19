import os
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
MODEL_TYPE = "text-generation"
# MODEL_TYPE = "question-answering"

env = os.getenv("APP_ENV")
if env == "prod":
    print("Running in production mode.")
    # LLM_MODEL = "Qwen/Qwen3-8B"
    LLM_MODEL = "FutureMa/Eva-4B"
else:
    print("Running in development mode.")
    LLM_MODEL = "Qwen/Qwen3-0.6B"
    # LLM_MODEL = "google/flan-t5-base"

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

auto_model = AutoModelForCausalLM if MODEL_TYPE == "text-generation" else AutoModelForQuestionAnswering
llm_model = auto_model.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    dtype="auto",
    trust_remote_code=True
)

qa_pipeline = pipeline(
    MODEL_TYPE,
    model=llm_model,
    tokenizer=tokenizer,
    return_full_text=False   # 只要答案部分
)

def get_embedder():
    return embed_fn

def get_pipeline():
    return qa_pipeline