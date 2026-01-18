import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
model_type = "text-generation"

env = os.getenv("APP_ENV")
if env == "prod":
    print("Running in production mode.")
    LLM_MODEL = "Qwen/Qwen3-8B"
else:
    print("Running in development mode.")
    LLM_MODEL = "Qwen/Qwen3-0.6B"

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

_model = SentenceTransformer(EMBED_MODEL)

def get_embedder():
    # chromadb 要求係一個 function，輸入 list[str] → 輸出 list[list[float]]
    def embed_fn(texts: list[str]) -> list[list[float]]:
        return _model.encode(texts).tolist()
    return embed_fn

def get_pipeline():
    return qa_pipeline