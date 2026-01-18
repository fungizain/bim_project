from pathlib import Path
import shutil

if Path("src").exists():
    BASE_PATH = Path("src")
else:
    BASE_PATH = Path(".")

UPLOAD_PATH = BASE_PATH / "upload_pdfs"
CHROMA_PATH = BASE_PATH / "chroma_db"
OUTPUT_PATH = BASE_PATH / "output_files"

ALL_PATHS = [UPLOAD_PATH, CHROMA_PATH, OUTPUT_PATH]

for p in ALL_PATHS:
    p.mkdir(parents=True, exist_ok=True)

def reset_folders():
    for folder in [UPLOAD_PATH, OUTPUT_PATH]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    return "✅ Reset success"