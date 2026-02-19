import hashlib
import os
from pathlib import Path
from docling.chunking import HybridChunker, DocChunk
from docling.document_converter import DocumentConverter
from fastapi import UploadFile
from langchain_core.documents import Document
from transformers import AutoTokenizer

from src.config import SPECIFIC_UPLOAD_PATH, SHARED_UPLOAD_PATH

env = os.getenv("APP_ENV")
if env == "prod":
    cache_path = "/tmp/torch_cache"
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    os.environ["PYTORCH_KERNEL_CACHE_PATH"] = cache_path

MODEL_NAME = "bert-base-uncased"
MAX_TOKENS = 1024

converter = DocumentConverter()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_length=MAX_TOKENS,
    truncation_side="right"
)
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True
)

def parse_chunk(chunk: DocChunk) -> Document:
    chunk_id = hashlib.md5(chunk.text.encode("utf-8")).hexdigest()
    page_numbers = [
        page_no
        for page_no in sorted(
            set(
                prov.page_no
                for item in chunk.meta.doc_items
                for prov in item.prov
            )
        )
    ]

    debug_file = Path("debug.txt")
    with debug_file.open("a", encoding="utf-8") as f:  # 用 append 模式
        f.write(f"--- Chunk {chunk_id} ---\n")
        f.write(chunk.text.strip() + "\n\n")

    return Document(
        page_content=chunk.text,
        metadata={
            "source": chunk.meta.origin.filename,
            "chunk_id": chunk_id,
            "pages": str(page_numbers)
        }
    )

def load_file(file_path: Path) -> list[Document]:
    result = converter.convert(file_path)
    chunk_iter = chunker.chunk(dl_doc=result.document)
    chunks = list(chunk_iter)
    chunks = [parse_chunk(chunk) for chunk in chunks]
    return chunks

def process_uploaded(upload_file, path: Path) -> list[Document]:
    try:
        file_path = path / upload_file.filename
        with open(file_path, "wb") as f:
            f.write(upload_file.file.read())

        documents = load_file(file_path)
        return documents
    except Exception as e:
        print(f"Error processing uploaded PDF: {str(e)}")
        return []
    
def process_specific_upload(upload_file: UploadFile) -> list[Document]:
    return process_uploaded(upload_file, SPECIFIC_UPLOAD_PATH)

def process_shared_upload(upload_file: UploadFile) -> list[Document]:
    return process_uploaded(upload_file, SHARED_UPLOAD_PATH)