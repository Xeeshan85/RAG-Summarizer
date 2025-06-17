import re
from datetime import datetime, timedelta
import markdown

import requests
import feedparser
from bs4 import BeautifulSoup
import time

class NewsRetriever:
    def __init__(self):
        ## free RSS feeds
        self.news_sources = {
            'BBC World': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'Reuters': 'http://feeds.reuters.com/Reuters/worldNews',
            'CNN World': 'http://rss.cnn.com/rss/edition.rss',
            'AP News': 'https://feeds.apnews.com/ApNews/apf-topnews',
            'Al Jazeera': 'https://www.aljazeera.com/xml/rss/all.xml',
            'NPR World': 'https://feeds.npr.org/1004/feed.json',
            'Guardian World': 'https://www.theguardian.com/world/rss'
        }
        
    def fetch_rss_news(self, source_name, url, max_articles=5):
        try:
            print(f"Fetching from {source_name}...")
            feed = feedparser.parse(url)
            articles = []
            for entry in feed.entries[:max_articles]:
                article = {
                    'source': source_name,
                    'title': entry.get('title', 'No Title'),
                    'summary': entry.get('summary', entry.get('description', 'No summary available')),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', str(datetime.now())),
                    'content': self._extract_content(entry.get('link', ''))
                }
                articles.append(article)
                time.sleep(1)
            
            return articles
        except Exception as e:
            print(f"Error fetching from {source_name}: {e}")
            return []
    
    def _extract_content(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            ## Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            ## Try to find main content
            content_selectors = [
                'article', '.article-content', '.story-body', 
                '.entry-content', '.post-content', 'main'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text()
                    break
            
            if not content:
                content = soup.get_text()
            
            content = re.sub(r'\s+', ' ', content).strip()
            return content[:4000] # Limiting len
            
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return ""
    
    def fetch_all_news(self, articles_per_source=3):
        all_articles = []
        for source_name, url in self.news_sources.items():
            articles = self.fetch_rss_news(source_name, url, articles_per_source)
            all_articles.extend(articles)
            print(f"Retrieved {len(articles)} articles from {source_name}")
        
        return all_articles
