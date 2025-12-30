from pathlib import Path
import fitz
import ocrmypdf
import subprocess
import shutil
import tabula
from fastapi import UploadFile
import pandas as pd

UPLOAD_FOLDER = Path("src/upload_pdfs")
OUTPUT_PDF_FOLDER = Path("src/output_pdfs")
TEXT_FOLDER = Path("src/output_texts")

for p in [UPLOAD_FOLDER, OUTPUT_PDF_FOLDER, TEXT_FOLDER]:
    p.mkdir(parents=True, exist_ok=True)

def ensure_text_layer(input_pdf: Path, output_pdf: Path, lang="eng"):
    """確保 PDF 有文字層，否則 OCR + Ghostscript fallback"""
    try:
        doc = fitz.open(input_pdf)
        text = doc[0].get_text()
        doc.close()
        if text.strip():
            return input_pdf
        try:
            ocrmypdf.ocr(str(input_pdf), str(output_pdf), language=lang, skip_text=True)
            return output_pdf
        except Exception:
            fixed_pdf = input_pdf.parent / f"fixed_{input_pdf.name}"
            subprocess.run([
                "gs", "-o", str(fixed_pdf),
                "-sDEVICE=pdfwrite",
                "-dPDFSETTINGS=/prepress",
                str(input_pdf)
            ], check=True)
            ocrmypdf.ocr(str(fixed_pdf), str(output_pdf), language=lang, skip_text=True)
            return output_pdf
    except Exception as e:
        raise RuntimeError(f"PDF 無法處理: {e}")

def extract_page_content(pdf_path: Path):
    """用 fitz 抽取每頁文字"""
    results = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        results.append({
            "page": page_num + 1,
            "type": "Text",
            "content": text.strip()
        })
    doc.close()
    return results

def extract_tables_tabula(pdf_path: Path):
    """用 Tabula 抽取表格，轉成 flatten text"""
    dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)
    table_texts = []
    if dfs and len(dfs) > 0:
        for idx, df in enumerate(dfs, start=1):
            # 每個 DataFrame flatten 成文字
            lines = []
            for i, row in df.iterrows():
                line = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                lines.append(line)
            table_text = f"[Table {idx}]\n" + "\n".join(lines)
            table_texts.append(table_text)
    return table_texts

def process_pdf(input_pdf: Path,
                output_pdf_folder: Path = OUTPUT_PDF_FOLDER,
                output_text_folder: Path = TEXT_FOLDER,
                lang="eng"):
    """主流程：OCR → fitz抽文字 → Tabula抽表格 → 合併輸出"""
    try:
        output_txt = output_text_folder / f"{input_pdf.stem}.txt"
        if output_txt.exists():
            return None, None, output_txt

        output_pdf = output_pdf_folder / f"ocr_{input_pdf.name}"
        pdf_with_text = ensure_text_layer(input_pdf, output_pdf, lang=lang)

        # 抽文字
        page_contents = extract_page_content(pdf_with_text)
        full_text = "\n\n".join(
            [f"[Page {c['page']} - {c['type']}]\n{c['content']}" for c in page_contents]
        )

        # 抽表格 → flatten 入文字
        # table_texts = extract_tables_tabula(pdf_with_text)
        # if table_texts:
        #     full_text += "\n\n" + "\n\n".join(table_texts)

        # 寫入 txt
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(full_text)

        return full_text, output_pdf, output_txt
    except Exception as e:
        return None, None, None

def process_uploaded_pdf(file: UploadFile,
                         output_pdf_folder: Path = OUTPUT_PDF_FOLDER,
                         output_text_folder: Path = TEXT_FOLDER,
                         lang="eng"):
    pdf_path = UPLOAD_FOLDER / file.filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return process_pdf(pdf_path, output_pdf_folder, output_text_folder, lang)