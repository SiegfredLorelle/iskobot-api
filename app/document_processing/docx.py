from docx import Document
from typing import Dict, Any
from ..utils.text_cleaner import clean_text

def extract_text_from_docx(blob) -> Dict[str, Any]:
    """Extract text from DOCX with metadata."""
    print(f"Processing: {blob.name}")
    metadata = {
        "source": blob.name,
        "sections": []
    }
    
    with blob.open("rb") as f:
        doc = Document(f)
        paragraphs = []
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(clean_text(text))
        
        metadata["total_paragraphs"] = len(paragraphs)
    
    return {
        "text": "\n".join(paragraphs),
        "metadata": metadata
    }
