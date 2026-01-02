from pathlib import Path
import shutil

UPLOAD_FOLDER = Path("src/upload_pdfs")
OUTPUT_PDF_FOLDER = Path("src/output_pdfs")
TEXT_FOLDER = Path("src/output_texts")
FAISS_STORE = Path("src/faiss_store")

ALL_FOLDERS = [UPLOAD_FOLDER, OUTPUT_PDF_FOLDER, TEXT_FOLDER, FAISS_STORE]

for p in ALL_FOLDERS:
    p.mkdir(parents=True, exist_ok=True)

def reset_folders():
    for folder in ALL_FOLDERS:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    return "✅ Reset success"