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

    # basic unescape + normalize whitespace
    s = unescape(s)
    s = s.replace("\u00A0", " ")  # non-breaking space
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # normalize dashes
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip()

    # remove trailing/leading pipes, stray underscores and repeated punctuation
    s = re.sub(r"^[\|\_\-\s]+", "", s)   # leading noise
    s = re.sub(r"[\|\_\-\s]+$", "", s)   # trailing noise

    # remove isolated OCR artifacts like "—_ 3" -> "3" or "- 7" -> "7"
    s = re.sub(r"^[\-\—\_]+\s*", "", s)  # leading dashes/underscores
    s = re.sub(r"\s*[\|\_]+\s*", " ", s) # internal pipes/underscores -> space

    # fix stray sequences like "<12000-" or "< 12000-" -> "<12000"
    s = re.sub(r"<\s*([0-9]+)[\-\s]*", r"<\1", s)

    # remove stray single-letter prefixes like "L :" or "A :" at line start (common OCR noise)
    s = re.sub(r"^[A-Za-z]\s*[:\-]\s*", "", s)

    # normalize slashes, plus signs, commas and semicolons spacing
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s*;\s*", "; ", s)

    # collapse repeated punctuation (e.g., "——" or "--" -> "-")
    s = re.sub(r"[-]{2,}", "-", s)
    s = re.sub(r"[;]{2,}", ";", s)

    # remove stray braces or weird leading tokens like "{| _" -> ""
    s = re.sub(r"[\{\}\[\]\|_]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # common OCR typos: "Vanne" -> "Vane", fix obvious misspellings if desired
    s = re.sub(r"\bVanne\b", "Vane", s, flags=re.IGNORECASE)

    # remove lone punctuation at ends
    s = re.sub(r"^[\:\-\;\,\.]+", "", s)
    s = re.sub(r"[\:\-\;\,\.]+$", "", s)

    # final trim and return
    return s.strip()

def table_to_matrix(table_tag) -> List[List[str]]:
    grid: List[List[Optional[str]]] = []
    rows = table_tag.find_all("tr")
    for r_idx, tr in enumerate(rows):
        while len(grid) <= r_idx:
            grid.append([])
        col_idx = 0
        for cell in tr.find_all(["td", "th"]):
            while col_idx < len(grid[r_idx]) and grid[r_idx][col_idx] is not None:
                col_idx += 1
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            text = clean_text(cell.get_text(separator=" ", strip=True))
            for dr in range(rowspan):
                target_r = r_idx + dr
                while len(grid) <= target_r:
                    grid.append([])
                while len(grid[target_r]) < col_idx + colspan:
                    grid[target_r].append(None)
                for dc in range(colspan):
                    if dr == 0 and dc == 0:
                        grid[target_r][col_idx + dc] = text
                    else:
                        grid[target_r][col_idx + dc] = ""
            col_idx += colspan
    max_cols = max((len(r) for r in grid), default=0)
    normalized = []
    for r in grid:
        newr = [(c if c is not None else "") for c in r]
        if len(newr) < max_cols:
            newr.extend([""] * (max_cols - len(newr)))
        normalized.append(newr)
    return normalized

def html_table_to_markdown_kv(html: str) -> str:
    """
    Parse HTML table(s) and return Markdown key-value output.
    - Each table becomes a "### Table N" section.
    - Rows where first column looks like a label become:
        - **Label**: Header2=val2; Header3=val3
    - Otherwise rows become:
        - **Row i**: Col1=val1; Col2=val2; ...
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return clean_text(soup.get_text(separator="\n", strip=True))

    parts = []
    for t_idx, table in enumerate(tables, start=1):
        matrix = table_to_matrix(table)
        if not matrix:
            continue

        first = matrix[0]
        header_likely = any(re.search(r"[A-Za-z\u4e00-\u9fff]", c) and not re.fullmatch(r"[\d\.\-\/\s]+", c) for c in first)
        if header_likely:
            headers = [clean_text(c) or f"Column {i+1}" for i, c in enumerate(first)]
            data_rows = matrix[1:]
        else:
            headers = [f"Column {i+1}" for i in range(len(first))]
            data_rows = matrix

        lines = [f"### Table {t_idx}"]
        for r_idx, row in enumerate(data_rows, start=1):
            row = [clean_text(c) for c in row]
            # build mapping
            row_map = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
            first_col_name = headers[0]
            first_col_val = row_map.get(first_col_name, "")
            # decide label-like: header name contains label keywords OR header is 'Description' etc.
            label_like_header = bool(re.search(r"(desc|description|field|item|name)", first_col_name.lower()))
            # also if first column value looks like a label (short, non-numeric)
            label_like_value = bool(first_col_val and re.search(r"[A-Za-z\u4e00-\u9fff]", first_col_val) and len(first_col_val) < 80)
            if (label_like_header or label_like_value) and len(headers) >= 2:
                label = first_col_val or "(no label)"
                kvs = []
                for k in headers[1:]:
                    v = row_map.get(k, "")
                    if v:
                        kvs.append(f"{k}={v}")
                if kvs:
                    lines.append(f"- **{label}**: " + "; ".join(kvs))
                else:
                    lines.append(f"- **{label}**:")
            else:
                kvs = [f"{k}={v}" for k, v in row_map.items() if v]
                if kvs:
                    lines.append(f"- **Row {r_idx}**: " + "; ".join(kvs))
                else:
                    # empty row fallback
                    lines.append(f"- **Row {r_idx}**: " + "; ".join([f"{k}=" for k in headers]))
        parts.append("\n".join(lines))
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
            result.append(html_table_to_markdown_kv(html_text))
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