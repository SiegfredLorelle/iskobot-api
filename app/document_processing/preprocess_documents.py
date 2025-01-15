from app.document_processing.docx import extract_text_from_docx
from app.document_processing.pdf import extract_text_from_pdf
from app.document_processing.pptx import extract_text_from_pptx
from typing import Dict, Any

def preprocess_document(blob, file_type) -> Dict[str, Any]:
    if file_type == "pdf":
        return extract_text_from_pdf(blob)
    elif file_type == "docx":
        return extract_text_from_docx(blob)
    elif file_type == "pptx":
        return extract_text_from_pptx(blob)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")