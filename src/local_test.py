from pathlib import Path
import ocrmypdf
import camelot
import tabula
from pdf_service import extract_page_content, extract_tables_tabula, process_pdf

def main():
    directory = Path("src/upload_pdfs/")
    filenames = ["Testing File Simple.pdf"]

    pdf_file = directory / filenames[0]
    
    # result = extract_page_content(pdf_file)
    # for page in result:
    #     print(f"{page['content']}\n{'-'*40}\n")

    # tables = extract_tables_tabula(pdf_file)
    # for t in tables:
    #     print(t)

    processed = process_pdf(pdf_file)

if __name__ == "__main__":
    main()