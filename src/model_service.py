from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
model_type = "text-generation"
# LLM_MODEL = "Qwen/Qwen3-0.6B"
LLM_MODEL = "tencent/WeDLM-8B-Instruct"

embedder = SentenceTransformer(EMBED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    device_map="auto",
    dtype="auto"
)

qa_pipeline = pipeline(
    model_type,
    model=llm_model,
    tokenizer=tokenizer,
    return_full_text=False   # 只要答案部分
)

def get_embedder():
    return embedder

def get_pipeline():
    return qa_pipeline
