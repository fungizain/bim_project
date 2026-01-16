from pathlib import Path
import shutil

UPLOAD_PATH = Path("src/upload_pdfs")
CHROMA_PATH = Path("src/chroma_db")
OUTPUT_PATH = Path("src/output_files")

ALL_PATHS = [UPLOAD_PATH, CHROMA_PATH, OUTPUT_PATH]

for p in ALL_PATHS:
    p.mkdir(parents=True, exist_ok=True)

def reset_folders():
    for folder in ALL_PATHS:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    return "✅ Reset success"