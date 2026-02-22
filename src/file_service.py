import hashlib
import os
from pathlib import Path
import shutil
from docling.chunking import HybridChunker, DocChunk
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode
)
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

pipeline_options = ThreadedPdfPipelineOptions(
    accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
    ocr_batch_size=4,
    layout_batch_size=64,
    table_batch_size=4,
    do_table_structure=True
)
pipeline_options.table_structure_options.do_cell_matching = False
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
pipeline_options.ocr_options = RapidOcrOptions(backend="torch")
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
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

    seen_ids = set()
    final_chunks: list[Document] = []

    for chunk in chunks:
        doc = parse_chunk(chunk)
        cid = doc.metadata["chunk_id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            final_chunks.append(doc)

    return final_chunks

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

def process_saved(file_path: str, path: Path) -> list[Document]:
    try:
        target_path = path / Path(file_path).name
        shutil.copy(file_path, target_path)
        documents = load_file(target_path)
        return documents
    except Exception as e:
        print(f"Error processing saved PDF: {str(e)}")
        return []

def process_specific_saved(file_path: str) -> list[Document]:
    return process_saved(file_path, SPECIFIC_UPLOAD_PATH)

def process_shared_saved(file_path: str) -> list[Document]:
    return process_saved(file_path, SHARED_UPLOAD_PATH)