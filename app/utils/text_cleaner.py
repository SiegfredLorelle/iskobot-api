import re

def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace and unwanted characters."""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    return text.strip()