from src.config import SPECIFIC_UPLOAD_PATH
from src.chroma_service import add_to_chroma
from src.pdf_service import load_pdf

def test_process_pdf():
    directory = SPECIFIC_UPLOAD_PATH
    filenames = ["Testing File Simple.pdf", "Test Case 1_E1_CHR.pdf", "Test Case 2_88-111.pdf"]

    for file in filenames:
        file_path = directory / file
        print(f"Processing file: {file_path}")
        document = load_pdf(file_path)
        add_to_chroma(document)

def main():
    test_process_pdf()

if __name__ == "__main__":
    main()