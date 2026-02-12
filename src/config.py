from pathlib import Path
import shutil

if Path("src").exists():
    BASE_PATH = Path("src")
else:
    BASE_PATH = Path(".")

SPECIFIC_UPLOAD_PATH = BASE_PATH / "specific_upload"
SHARED_UPLOAD_PATH = BASE_PATH / "shared_upload"
CHROMA_PATH = BASE_PATH / "chroma_db"
OUTPUT_PATH = BASE_PATH / "output_files"

ALL_PATHS = [SPECIFIC_UPLOAD_PATH, SHARED_UPLOAD_PATH, CHROMA_PATH, OUTPUT_PATH]

for p in ALL_PATHS:
    p.mkdir(parents=True, exist_ok=True)

def list_folders(folder: Path):
    return [item.name for item in folder.iterdir()]

def list_specific_folders():
    return list_folders(SPECIFIC_UPLOAD_PATH)

def list_shared_folders():
    return list_folders(SHARED_UPLOAD_PATH)

def reset(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return "✅ Reset success"

def reset_specific_folders():
    return reset(SPECIFIC_UPLOAD_PATH)

def reset_shared_folders():
    return reset(SHARED_UPLOAD_PATH)