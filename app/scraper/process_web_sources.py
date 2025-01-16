from typing import List
from app.scraper.rag_web_scraper import RAGWebScraper
from langchain.schema import Document

def process_web_sources(urls: List[str], max_pages_per_site: int = 100) -> List[Document]:
    """Process web sources and return documents."""
    all_documents = []
    for url in urls:
        scraper = RAGWebScraper(url)
        # if not scraper.respect_robots_txt():
        #     print(f"Scraping not allowed for {url}")
        # else:
        documents = scraper.scrape_site(max_pages=max_pages_per_site)
        all_documents.extend(documents)
    return all_documents