from typing import Dict, Any
import fitz  # PyMuPDF
from ..utils.text_cleaner import clean_text

def extract_text_from_pdf(blob) -> Dict[str, Any]:
    """Extract text from PDF with metadata."""
    print(f"Processing: {blob.name}")
    metadata = {
        "source": blob.name,
        "page_numbers": []
    }
    
    with blob.open("rb") as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        text_by_page = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text with formatting details
            text = page.get_text("text")
            cleaned_text = clean_text(text)
            
            if cleaned_text.strip():  # Only add non-empty pages
                text_by_page.append(cleaned_text)
                metadata["page_numbers"].append(page_num + 1)
        
        doc.close()
    
    return {
        "text": "\n".join(text_by_page),
        "metadata": metadata
    }