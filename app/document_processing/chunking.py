from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def create_chunks(text: str, metadata: dict) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        chunk_size=1_000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )

    return text_splitter.create_documents(
        texts=[text],
        metadatas=[metadata]
    )