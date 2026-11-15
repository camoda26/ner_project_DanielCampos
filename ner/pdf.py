from pathlib import Path
import pymupdf


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    doc = pymupdf.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()
    return text.strip()


def extract_pdf_dataset(pdf_dir: str | Path, dataset: list[dict]) -> list[dict]:
    pdf_dir = Path(pdf_dir)
    results = []

    for item in dataset:
        doc_id = item["id"]
        pdf_path = pdf_dir / f"{doc_id}.pdf"

        if not pdf_path.exists():
            pdf_path = pdf_dir / f"{doc_id}_scanned.pdf"

        if not pdf_path.exists():
            continue

        text = extract_text_from_pdf(pdf_path)

        results.append({
            "id": doc_id,
            "text": text
        })

    return results
