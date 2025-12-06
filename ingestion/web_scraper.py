"""Web scraping for knowledge base"""
import asyncio
import aiohttp
import uuid
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

class WebScraper:
    """Async web scraper with depth control"""
    
    def __init__(self, max_depth: int = 2, max_pages: int = 30):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited: Set[str] = set()
        self.chunks: List[Dict] = []
        
    async def scrape_url(
        self,
        url: str,
        session: aiohttp.ClientSession,
        depth: int = 0
    ):
        """Recursively scrape URL"""
        if (
            depth > self.max_depth
            or len(self.visited) >= self.max_pages
            or url in self.visited
        ):
            return
        
        self.visited.add(url)
        
        try:
            async with session.get(url, timeout=10, allow_redirects=False) as response:
                if response.status >= 400:
                    print(f"⚠️ Skipping {url} (status {response.status})")
                    return
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract text content
                text_elements = soup.find_all(['p', 'div', 'li', 'h1', 'h2', 'h3'])
                text = " ".join([el.get_text() for el in text_elements])
                
                # Chunk text
                words = text.split()
                chunk_size = 500
                overlap = 50
                
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_text = " ".join(words[i:i + chunk_size]).strip()
                    
                    if len(chunk_text) > 100:
                        self.chunks.append({
                            "id": str(uuid.uuid4()),
                            "text": chunk_text,
                            "page_number": f"web-{urlparse(url).path}",
                            "source": url,
                            "type": "web"
                        })
                
                print(f"✅ Scraped {url} (depth {depth})")
                
                # Extract links for recursive scraping
                if depth < self.max_depth:
                    base_domain = urlparse(url).netloc
                    links = soup.find_all('a', href=True)
                    
                    tasks = []
                    for link in links:
                        abs_url = urljoin(url, link['href'])
                        # Only follow same-domain links
                        if urlparse(abs_url).netloc == base_domain:
                            if abs_url not in self.visited:
                                tasks.append(
                                    self.scrape_url(abs_url, session, depth + 1)
                                )
                    
                    # Limit concurrent requests
                    if tasks:
                        await asyncio.gather(*tasks[:5])
                        
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
    
    async def scrape(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs"""
        async with aiohttp.ClientSession() as session:
            tasks = [self.scrape_url(url, session) for url in urls]
            await asyncio.gather(*tasks)
        
        print(f"✅ Total scraped chunks: {len(self.chunks)}")
        return self.chunks
