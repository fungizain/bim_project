from pathlib import Path
import json
import pdfplumber
import ocrmypdf
import subprocess
import shutil
from fastapi import UploadFile

from src.folder_service import OUTPUT_PDF_FOLDER, UPLOAD_FOLDER, TEXT_FOLDER


def has_text_layer(pdf_path: Path) -> bool:
    """檢查 PDF 第一頁是否有文字層"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page_text = pdf.pages[0].extract_text() or ""
        return bool(first_page_text.strip())
    except Exception:
        return False


def run_ocr(input_pdf: Path, output_pdf: Path, lang: str = "eng") -> Path:
    """對 PDF 進行 OCR"""
    ocrmypdf.ocr(str(input_pdf), str(output_pdf), language=lang, skip_text=True)
    return output_pdf


def fix_pdf_with_ghostscript(input_pdf: Path) -> Path:
    """用 Ghostscript 修復 PDF"""
    fixed_pdf = input_pdf.parent / f"fixed_{input_pdf.name}"
    subprocess.run([
        "gs", "-o", str(fixed_pdf),
        "-sDEVICE=pdfwrite",
        "-dPDFSETTINGS=/prepress",
        str(input_pdf)
    ], check=True)
    return fixed_pdf


def extract_text_pages(pdf_path: Path) -> list[dict]:
    """逐頁提取文字，返回 JSON 格式 [{page, text}]"""
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):  # page number 從 1 開始
            page_text = page.extract_text()
            if page_text:
                results.append({
                    "filename": pdf_path.name,
                    "page": page_num,
                    "text": page_text.strip()
                })
    return results


def extract_text_from_pdf(input_pdf: Path,
                          output_pdf_folder: Path = OUTPUT_PDF_FOLDER,
                          lang: str = "eng") -> list[dict]:
    """
    主流程：
    1. 檢查 PDF 是否有文字層
    2. 如果冇 → OCR
    3. OCR 失敗 → Ghostscript 修復再 OCR
    4. 提取文字
    """
    try:
        output_pdf = output_pdf_folder / f"ocr_{input_pdf.name}"

        if has_text_layer(input_pdf):
            pdf_to_read = input_pdf
        else:
            try:
                pdf_to_read = run_ocr(input_pdf, output_pdf, lang)
            except Exception:
                fixed_pdf = fix_pdf_with_ghostscript(input_pdf)
                pdf_to_read = run_ocr(fixed_pdf, output_pdf, lang)

        return extract_text_pages(pdf_to_read)

    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def process_pdf(input_pdf: Path,
                output_pdf_folder: Path = OUTPUT_PDF_FOLDER,
                output_text_folder: Path = TEXT_FOLDER,
                lang: str = "eng") -> Path | None:
    try:
        output_json = output_text_folder / f"{input_pdf.stem}.json"
        if output_json.exists():
            print("JSON file already exists, skip writing.")
            return output_json

        # extract_text_from_pdf 要返回 [{page, text}, ...]
        pages = extract_text_from_pdf(input_pdf, output_pdf_folder, lang)
        if not pages:
            print("No text extracted from PDF.")
            return None

        output_text_folder.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)
        print("JSON file written successfully.")

        return output_json
    except Exception as e:
        print(f"Error in process_pdf: {e}")
        return None


def process_uploaded_pdf(file: UploadFile,
                         output_pdf_folder: Path = OUTPUT_PDF_FOLDER,
                         output_text_folder: Path = TEXT_FOLDER,
                         lang: str = "eng") -> Path | None:
    pdf_path = UPLOAD_FOLDER / file.filename
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return process_pdf(pdf_path, output_pdf_folder, output_text_folder, lang)