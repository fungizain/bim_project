import faiss, pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer

from src.pdf_service import TEXT_FOLDER

@dataclass
class DocChunk:
    file_name: str
    chunk_id: int
    text: str

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
MAX_CONTEXT_CHARS = 6000

FAISS_STORE = Path("src/faiss_store")
FAISS_STORE.mkdir(parents=True, exist_ok=True)

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
def update_faiss(embedder: SentenceTransformer):
    doc_chunks = []

    # 處理 txt files
    for file in TEXT_FOLDER.glob("*.txt"):
        content = file.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            doc_chunks.append(DocChunk(file_name=file.name, chunk_id=i, text=ch))

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
def search_chunks(query: str, index, embedder, doc_chunks: List[DocChunk], top_k: int = TOP_K) -> List[DocChunk]:
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    return [doc_chunks[i] for i in I[0]]

def build_context(chunks: List[DocChunk], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts, length = [], 0
    for c in chunks:
        header = f"[{c.file_name}#chunk-{c.chunk_id}]\n"
        candidate = header + c.text.strip() + "\n\n"
        if length + len(candidate) > max_chars:
            break
        parts.append(candidate)
        length += len(candidate)
    return "".join(parts)

# --------- main ---------
def process_and_update_index(
    text_file: Path,
    embedder: SentenceTransformer,
    lang: str = "eng"
):
    """
    主流程：
    1. 接收已經處理好嘅文字檔 (text_file)
    2. 更新 FAISS index
    """
    if text_file and text_file.exists():
        update_faiss(embedder)
        return {
            "status": "success",
            "text_file": text_file
        }

    return {
        "status": "failed",
        "text_file": None
    }

def prepare_prompt_from_query(
    query: str,
    embedder: SentenceTransformer,
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

    prompt = (
        "You are assisting a government department staff to retrieve information from official PDF documents.\n"
        "Use ONLY the provided context excerpts to answer.\n"
        "Do not add external knowledge. Do not repeat the context verbatim.\n"
        "The context may contain flattened tables extracted from PDFs.\n"
        "Your task is to read these flattened rows and reconstruct the requested information into a clear structured table.\n"
        "The table must follow exactly the columns requested in the question.\n"
        "If multiple rows match, include them all in the table.\n"
        "Give the closest possible answer from the table.\n"
        "If the answer is not present in the context, reply exactly: 'Not found in context.'\n\n"
        f"Context:\n{context}\n"
        f"Question: {query}\nAnswer (in table format):"
    )

    return prompt, hits