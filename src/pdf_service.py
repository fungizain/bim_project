import base64
import hashlib
import json
from pathlib import Path
import zlib
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from src.config import OUTPUT_PATH, UPLOAD_PATH

# Extract the contents of an orig_elements field.
def extract_orig_elements(orig_elements):
    decoded_orig_elements = base64.b64decode(orig_elements)
    decompressed_orig_elements = zlib.decompress(decoded_orig_elements)
    return decompressed_orig_elements.decode('utf-8')

def parse_chunk(chunk) -> Document:
    data = chunk.metadata.to_dict()
    filename = data["filename"]

    elements = extract_orig_elements(data["orig_elements"])
    json_obj = json.loads(elements)

    result = []
    pages = []
    for e in json_obj:
        pages.append(e["metadata"]["page_number"])

        if e["type"] == "Table":
            result.append(e["metadata"]["text_as_html"])
        elif e["type"] == "Image":
            continue # skip images for now
        else:
            result.append(e["text"])
    content = "\n".join(result)
    page_range = (min(pages), max(pages))
    chunk_id = hashlib.md5(content.encode("utf-8")).hexdigest()

    return Document(
        page_content=content,
        metadata={
            "source": filename,
            "chunk_id": chunk_id,
            "page_start": page_range[0],
            "page_end": page_range[1],
        }
    )

def load_pdf(file_path: Path) -> list[Document]:
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",                            # mandatory to infer tables
        infer_table_structure=True,                   # extract tables
        languages=["eng"],

        extract_images_in_pdf=True,                   # mandatory to set as ``True``
        extract_image_block_types=["Image"],          # Add 'Table' to list to extract image of tables
        extract_image_block_output_dir=OUTPUT_PATH,   # only works when ``extract_image_block_to_payload=False``
        extract_image_block_to_payload=True,          # if true, will extract base64 for API usage
    )

    chunks = chunk_by_title(
        elements,
        max_characters=2000,    # hard maximum
        new_after_n_chars=1000, # soft maximum
        overlap=200,            # default 0
    )

    pdf_chunks = []
    seen_ids = set()
    for chunk in chunks:
        doc = parse_chunk(chunk)
        cid = doc.metadata["chunk_id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            pdf_chunks.append(doc)
    return pdf_chunks

def process_uploaded_pdf(upload_file) -> list[Document]:
    try:
        file_path = UPLOAD_PATH / upload_file.filename
        with open(file_path, "wb") as f:
            f.write(upload_file.file.read())

        documents = load_pdf(file_path)
        return documents
    except Exception as e:
        print(f"Error processing uploaded PDF: {str(e)}")
        return []