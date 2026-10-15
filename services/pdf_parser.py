import fitz  # PyMuPDF
from typing import List, Dict, Any

class PDFParser:
    @staticmethod
    def extract_text_with_pages(pdf_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        extracted_pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                extracted_pages.append({
                    "page_number": page_num + 1,
                    "content": text.strip()
                })
        return extracted_pages