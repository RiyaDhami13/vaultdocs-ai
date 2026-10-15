from typing import List, Dict, Any

class ClauseSplitter:
    @staticmethod
    def split_into_chunks(pages: List[Dict[str, Any]], chunk_size: int = 600, overlap: int = 100) -> List[Dict[str, Any]]:
        chunks = []
        for page in pages:
            page_num = page["page_number"]
            text = page["content"]
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append({
                    "page_number": page_num,
                    "content": text[start:end]
                })
                start += chunk_size - overlap
        return chunks