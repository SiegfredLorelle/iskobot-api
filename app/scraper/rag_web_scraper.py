from bs4 import BeautifulSoup
import requests
import json
from datetime import datetime
from urllib.parse import urljoin, urlparse
import hashlib
from langchain.schema import Document
from typing import List, Optional, Dict
import time

class RAGWebScraper:
    def __init__(self, base_url: str, delay: float = 1.0):
        """
        Initialize the scraper with a base URL.
        
        Args:
            base_url: The starting URL to scrape
            delay: Time to wait between requests in seconds
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAGBot/1.0 (Educational Purpose)'
        })
        self.content_hashes: Dict[str, str] = {}  # Store content hashes to detect duplicates

    def get_content_hash(self, content: str) -> str:
        """Generate a hash of the content for deduplication."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def is_duplicate_content(self, content: str) -> Optional[str]:
        """
        Check if content is duplicate and return the original URL if it is.
        
        Args:
            content: The content to check
            
        Returns:
            The URL of the duplicate content if found, None otherwise
        """
        content_hash = self.get_content_hash(content)
        return self.content_hashes.get(content_hash)

    def get_page_content(self, url: str) -> Optional[str]:
        """Fetch and parse a single page."""
        try:
            time.sleep(self.delay)  # Rate limiting
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_page(self, html_content: str, url: str) -> Optional[Document]:
        """Parse HTML content and return a Document object."""
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer']):
            element.decompose()

        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if not main_content:
            return None

        # Get cleaned text content
        cleaned_content = main_content.get_text(separator=' ', strip=True)
        
        # Check for duplicate content
        duplicate_url = self.is_duplicate_content(cleaned_content)
        if duplicate_url:
            print(f"Duplicate content found: {url} matches {duplicate_url}")
            return None

        # Store content hash
        content_hash = self.get_content_hash(cleaned_content)
        self.content_hashes[content_hash] = url

        # Extract metadata
        title = soup.title.string if soup.title else ''
        meta_description = soup.find('meta', {'name': 'description'})
        description = meta_description['content'] if meta_description else ''

        # Create metadata dictionary
        metadata = {
            'source': url,
            'title': title.strip() if title else '',
            'description': description.strip(),
            'source_type': 'web',
            'scraped_date': datetime.now().isoformat(),
            'domain': urlparse(url).netloc,
            'file_type': 'html',
            'content_hash': content_hash,
            'canonical_url': self.get_canonical_url(soup, url)
        }

        # Create Document object
        return Document(
            page_content=cleaned_content,
            metadata=metadata
        )

    def get_canonical_url(self, soup: BeautifulSoup, default_url: str) -> str:
        """Extract canonical URL if available."""
        canonical_tag = soup.find('link', {'rel': 'canonical'})
        return canonical_tag['href'] if canonical_tag else default_url

    def scrape_site(self, max_pages: int = 10) -> List[Document]:
        """
        Scrape the website and return list of Document objects.
        
        Args:
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of Document objects
        """
        visited_urls = set()
        documents = []
        urls_to_visit = [self.base_url]

        while urls_to_visit and len(visited_urls) < max_pages:
            url = urls_to_visit.pop(0)
            
            # Skip if URL was already visited
            if url in visited_urls:
                continue

            # Normalize URL
            parsed_url = urlparse(url)
            normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            if normalized_url in visited_urls:
                continue

            print(f"Scraping: {url}")
            html_content = self.get_page_content(url)
            if html_content:
                document = self.parse_page(html_content, url)
                if document:
                    documents.append(document)
                
                # Find more links to scrape
                soup = BeautifulSoup(html_content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(self.base_url, link['href'])
                    if (next_url.startswith(self.base_url) and 
                        next_url not in visited_urls and 
                        next_url not in urls_to_visit):
                        urls_to_visit.append(next_url)

            visited_urls.add(url)
            visited_urls.add(normalized_url)

        print(f"Scraped {len(documents)} unique documents")
        return documents

    def respect_robots_txt(self) -> bool:
        """Check robots.txt for crawling permissions."""
        robots_url = urljoin(self.base_url, '/robots.txt')
        try:
            response = requests.get(robots_url)
            return 'Disallow: /' not in response.text
        except:
            return True