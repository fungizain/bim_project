import base64
import hashlib
from html import unescape
import json
from pathlib import Path
import re
from typing import List, Optional
import zlib
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from src.config import OUTPUT_PATH, UPLOAD_PATH

# Extract the contents of an orig_elements field.
def extract_orig_elements(orig_elements):
    decoded_orig_elements = base64.b64decode(orig_elements)
    decompressed_orig_elements = zlib.decompress(decoded_orig_elements)
    return decompressed_orig_elements.decode('utf-8')

def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\|\s*$", "", s)  # trailing pipe noise
    return s

def table_to_matrix(table_tag) -> List[List[str]]:
    """
    Expand a <table> into a 2D matrix of strings, handling rowspan/colspan.
    Empty cells become empty strings.
    """
    grid: List[List[Optional[str]]] = []
    rows = table_tag.find_all("tr")
    for r_idx, tr in enumerate(rows):
        # ensure grid has row r_idx
        while len(grid) <= r_idx:
            grid.append([])
        col_idx = 0
        for cell in tr.find_all(["td", "th"]):
            # find next free column
            while col_idx < len(grid[r_idx]) and grid[r_idx][col_idx] is not None:
                col_idx += 1
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            text = clean_text(cell.get_text(separator=" ", strip=True))
            for dr in range(rowspan):
                target_r = r_idx + dr
                while len(grid) <= target_r:
                    grid.append([])
                # ensure target row long enough
                while len(grid[target_r]) < col_idx + colspan:
                    grid[target_r].append(None)
                for dc in range(colspan):
                    if dr == 0 and dc == 0:
                        grid[target_r][col_idx + dc] = text
                    else:
                        grid[target_r][col_idx + dc] = ""  # placeholder for spanned cell
            col_idx += colspan
    # normalize None -> ""
    max_cols = max((len(r) for r in grid), default=0)
    normalized = []
    for r in grid:
        newr = [(c if c is not None else "") for c in r]
        if len(newr) < max_cols:
            newr.extend([""] * (max_cols - len(newr)))
        normalized.append(newr)
    return normalized

def matrix_to_plain_text(matrix: List[List[str]]) -> str:
    """
    Convert matrix to plain text. Heuristics:
    - If first row looks like header, use it as column names.
    - If first column looks like labels (contains 'desc'/'description'/'field'/'item'/'name'),
      format rows as "Label: Col2=...; Col3=...".
    - Otherwise format each row as "Col1=...; Col2=...".
    """
    if not matrix:
        return ""
    first = matrix[0]
    # detect header
    header_likely = any(re.search(r"[A-Za-z\u4e00-\u9fff]", c) and not re.fullmatch(r"[\d\.\-\/\s]+", c) for c in first)
    if header_likely:
        headers = [clean_text(c) or f"Column {i+1}" for i, c in enumerate(first)]
        data_rows = matrix[1:]
    else:
        headers = [f"Column {i+1}" for i in range(len(first))]
        data_rows = matrix

    lines = []
    for row in data_rows:
        row = [clean_text(c) for c in row]
        # build dict
        row_dict = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
        first_col_name = headers[0]
        first_col_val = row_dict.get(first_col_name, "")
        label_like = bool(re.search(r"(desc|description|field|item|name)", first_col_name.lower()))
        if label_like and len(headers) >= 2:
            label = first_col_val or "(no label)"
            kvs = []
            for k in headers[1:]:
                v = row_dict.get(k, "")
                if v:
                    kvs.append(f"{k}={v}")
            if kvs:
                lines.append(f"{label}: " + "; ".join(kvs))
            else:
                lines.append(f"{label}:")
        else:
            kvs = [f"{k}={v}" for k, v in row_dict.items() if v]
            if kvs:
                lines.append("; ".join(kvs))
            else:
                # if all empty, still show columns
                lines.append("; ".join([f"{k}=" for k in headers]))
    return "\n".join(lines)

def html_table_to_plain_text(html: str) -> str:
    """
    Parse HTML (one or more tables) and return a plain-text representation suitable for general LLMs.
    - Returns a multi-line string.
    - If no <table> present, returns cleaned plain text of the input.
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        # fallback: return cleaned text of whole html
        return clean_text(soup.get_text(separator="\n", strip=True))

    parts = []
    for idx, table in enumerate(tables, start=1):
        matrix = table_to_matrix(table)
        if not matrix:
            continue
        text = matrix_to_plain_text(matrix)
        if text:
            parts.append(f"--- Table {idx} ---\n{text}")
    return "\n\n".join(parts)

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
            html_text = e["metadata"]["text_as_html"]
            result.append(html_table_to_plain_text(html_text))
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