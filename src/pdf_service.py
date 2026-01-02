import json
import pandas as pd
from pathlib import Path
import fitz
import ocrmypdf
import subprocess
import shutil
import tabula
from fastapi import UploadFile

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
    """用 Tabula 抽取表格，智能判斷是否繼承上一頁 column，並輸出 JSON"""
    dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)
    json_records = []
    last_columns = None

    if dfs and len(dfs) > 0:
        for idx, df in enumerate(dfs, start=1):
            # 判斷是否需要繼承上一頁 column
            if last_columns is not None and any("Unnamed" in str(c) for c in df.columns):
                if len(df.columns) == len(last_columns):
                    df.columns = last_columns
                else:
                    # 如果長度唔對，就只取前兩個 column
                    df = df.iloc[:, :len(last_columns)]
                    df.columns = last_columns
            else:
                last_columns = df.columns

            # flatten DataFrame → JSON
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    val = row[col]
                    if pd.notna(val):
                        record[col] = str(val).strip()
                    else:
                        record[col] = ""
                if record:  # 避免空行
                    json_records.append(record)

    return json_records

def process_pdf(input_pdf: Path,
                output_pdf_folder: Path = OUTPUT_PDF_FOLDER,
                output_text_folder: Path = TEXT_FOLDER,
                lang="eng"):
    """主流程：OCR → Tabula抽表格 → fitz抽文字 (fallback) → 合併輸出"""
    try:
        output_txt = output_text_folder / f"{input_pdf.stem}.txt"
        print("Output txt path:", output_txt)

        if output_txt.exists():
            print("Txt file already exists, skip writing.")
            return None, None, output_txt

        output_pdf = output_pdf_folder / f"ocr_{input_pdf.name}"
        pdf_with_text = ensure_text_layer(input_pdf, output_pdf, lang=lang)

        # 先試抽表格
        table_json = extract_tables_tabula(pdf_with_text)

        if table_json and len(table_json) > 0:
            full_text = "[Tables Extracted]\n" + json.dumps(table_json, indent=2, ensure_ascii=False)
        else:
            page_contents = extract_page_content(pdf_with_text)
            full_text = "\n\n".join(
                [f"[Page {c['page']} - {c['type']}]\n{c['content']}" for c in page_contents]
            )

        print("Full text length:", len(full_text))

        # 寫入 txt
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(full_text)
        print("Txt file written successfully.")

        return full_text, output_pdf, output_txt
    except Exception as e:
        print(f"Error in process_pdf: {e}")
        return None, None, None

def process_uploaded_pdf(file: UploadFile,
                         output_pdf_folder: Path = OUTPUT_PDF_FOLDER,
                         output_text_folder: Path = TEXT_FOLDER,
                         lang="eng"):
    pdf_path = UPLOAD_FOLDER / file.filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return process_pdf(pdf_path, output_pdf_folder, output_text_folder, lang)