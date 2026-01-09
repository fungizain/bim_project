from pathlib import Path
from pdf_service import process_pdf

def test_process_pdf():
    directory = Path("src/upload_pdfs/")
    filenames = ["Testing File Simple.pdf"]

    pdf_file = directory / filenames[0]
    text = process_pdf(pdf_file)
    print(text)

def main():
    test_process_pdf()

if __name__ == "__main__":
    main()