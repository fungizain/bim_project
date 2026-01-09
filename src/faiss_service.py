import json
import faiss
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from src.pdf_service import TEXT_FOLDER
from src.folder_service import FAISS_STORE

@dataclass
class DocChunk:
    filename: str
    page: int
    chunk_id: int
    text: str

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
MAX_CONTEXT_CHARS = 6000

# --------- Cache ---------
FAISS_CACHE = {
    "index": None,
    "doc_chunks": None
}

# --------- Chunking ---------
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# --------- FAISS ---------
def build_faiss_index(doc_chunks: List[DocChunk], embedder: SentenceTransformer):
    texts = [c.text for c in doc_chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def save_faiss_index(index, doc_chunks, embed_model_name, save_dir: Path):
    faiss.write_index(index, str(save_dir / "faiss.index"))
    meta = {"doc_chunks": doc_chunks, "embed_model_name": embed_model_name}
    with open(save_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

def load_faiss_index(save_dir: Path):
    index = faiss.read_index(str(save_dir / "faiss.index"))
    with open(save_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return index, meta["doc_chunks"]

# --------- Update FAISS ---------
def update_faiss(json_file: Path, embedder: SentenceTransformer):
    doc_chunks = []

    # 確認 JSON 存在
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    
    # 讀 JSON
    with open(json_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # 處理每一頁
    for i, entry in enumerate(pages):
        doc_chunks.append(
            DocChunk(
                filename=entry.get("filename"),
                page=entry.get("page"),
                chunk_id=i,
                text=entry.get("text", "").strip()
            )
        )

    # 建立 FAISS index
    if not doc_chunks:
        raise RuntimeError("No documents found to build FAISS index.")

    index = build_faiss_index(doc_chunks, embedder)
    save_faiss_index(index, doc_chunks, embedder.__class__.__name__, FAISS_STORE)

    # 更新 cache
    FAISS_CACHE["index"] = index
    FAISS_CACHE["doc_chunks"] = doc_chunks

# --------- Cache Access ---------
def get_faiss_index():
    """Lazy load FAISS index, return cached if available."""
    if FAISS_CACHE["index"] is None or FAISS_CACHE["doc_chunks"] is None:
        index, doc_chunks = load_faiss_index(FAISS_STORE)
        FAISS_CACHE["index"] = index
        FAISS_CACHE["doc_chunks"] = doc_chunks
    return FAISS_CACHE["index"], FAISS_CACHE["doc_chunks"]

# --------- Search ---------
def search_chunks(
    query: str,
    index,
    embedder,
    doc_chunks: List[DocChunk],
    top_k: int = TOP_K
) -> List[DocChunk]:
    """用 FAISS 檢索 top-k chunks，返回 DocChunk list"""
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    return [doc_chunks[i] for i in I[0] if i < len(doc_chunks)]


def build_context(chunks: List[DocChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """將檢索到嘅 chunks 拼成 context，包含 file_name + page + chunk_id"""
    parts, length = [], 0
    for c in chunks:
        header = f"[{c.filename}#page-{c.page}#chunk-{c.chunk_id}]\n"
        candidate = header + c.text.strip() + "\n\n"
        if length + len(candidate) > max_chars:
            break
        parts.append(candidate)
        length += len(candidate)
    return "".join(parts)

# --------- main ---------
def process_and_update_index(json_file: Path, embedder: SentenceTransformer):
    """
    主流程：
    1. 接收已經處理好嘅 JSON file
    2. 更新 FAISS index
    """
    if json_file and json_file.exists():
        update_faiss(json_file, embedder)
        return { "status": "success", "file": json_file }
    else:
        return { "status": "failed", "file": None }

def prepare_prompt_from_query(
    query: str,
    embedder: SentenceTransformer,
    prompt_template: str,
    top_k: int = TOP_K
) -> Tuple[str, list]:
    """
    1. 用 cache 或 lazy load FAISS index
    2. Search top-k chunks
    3. Build context
    4. Build QA prompt
    Return: (prompt, hits)
    """
    index, doc_chunks = get_faiss_index()
    if index is None:
        return "Index not ready. Please upload PDF first.", []

    hits = search_chunks(query, index, embedder, doc_chunks, top_k=top_k)
    context = build_context(hits)

    prompt = prompt_template.format(context=context, query=query)
    return prompt, hits
