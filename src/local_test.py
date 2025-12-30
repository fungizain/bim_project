from pathlib import Path
import ocrmypdf
import camelot
import tabula

JAVA_TOOL_OPTIONS=--enable-native-access=ALL-UNNAMED # for tabula

def run_ocr(pdf_file: Path, ocr_pdf: Path, lang="eng") -> Path:
    """Step 1: OCR → 確保 PDF 有文字層"""
    if not ocr_pdf.exists():
        try:
            ocrmypdf.ocr(
                str(pdf_file),
                str(ocr_pdf),
                language=lang,
                skip_text=True
            )
        except Exception as e:
            raise RuntimeError(f"OCR failed: {e}")
    return ocr_pdf

def extract_with_camelot(ocr_pdf: Path):
    """用 Camelot 抽表格"""
    try:
        tables = camelot.read_pdf(
            str(ocr_pdf),
            pages="all",
            flavor="lattice",
            suppress_stdout=True
        )
        for idx, table in enumerate(tables, start=1):
            csv_path = ocr_pdf.parent / f"camelot_csv_{idx}.csv"
            table.to_csv(str(csv_path))
    except Exception as e:
        raise RuntimeError(f"Camelot failed: {e}")

def extract_with_tabula(ocr_pdf: Path):
    """用 Tabula 抽表格"""
    try:
        # tabula 會直接返回 DataFrame list
        dfs = tabula.read_pdf(
            str(ocr_pdf),
            pages="all",
            multiple_tables=True
        )
        for idx, df in enumerate(dfs, start=1):
            csv_path = ocr_pdf.parent / f"tabula_csv_{idx}.csv"
            df.to_csv(csv_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Tabula failed: {e}")

def main():
    directory = Path("src/assets/test/")
    filenames = ["Test Case 1_E1_CHR.pdf"]

    pdf_file = directory / filenames[0]
    ocr_pdf = directory / f"ocr_{filenames[0]}"

    # Step 1: OCR
    ocr_pdf = run_ocr(pdf_file, ocr_pdf)
    # Step 2: 抽表格
    extract_with_tabula(ocr_pdf)

if __name__ == "__main__":
    main()