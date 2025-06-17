# RAG Document & News Summarizer

A comprehensive implementation of Retrieval-Augmented Generation (RAG) for document summarization with real-time news fetching and processing capabilities. The system combines semantic chunking, vector search, and automated news retrieval to keep you updated with the latest world news.

## 🚀 Features

### Document Processing
- **Multi-format support**: PDF, TXT, MD, JSON files
- **Semantic text chunking** with configurable overlap
- **Vector embeddings** using SentenceTransformers
- **Semantic retrieval** with ChromaDB vector database
- **AI-powered summarization** with BART model

### News Integration
- **Real-time news fetching** from multiple free sources (BBC, Reuters, CNN, AP News, Al Jazeera, NPR, Guardian)
- **Automated content extraction** from news URLs
- **Batch processing** of multiple articles
- **Smart summarization** tailored for news content
- **Multiple output formats** (JSON, Markdown, Console)

### Advanced Capabilities
- **Intelligent chunking** with sentence-aware splitting
- **Relevance scoring** for retrieved content
- **Compression tracking** (original vs summary length)
- **Error handling** and graceful degradation
- **Timestamped outputs** for tracking

## 📦 Setup

1. **Install requirements:**
```bash
pip install -r requirements.txt
```

2. **Additional dependencies for news features:**
```bash
pip install feedparser beautifulsoup4 requests nltk chromadb pypdf2
```


## 🎯 Usage

### Quick Start - News Summarization
```bash
python main.py
```

### Document Summarization
```python
from rag_summarizer import RAGSummarizer

# Initialize
rag = RAGSummarizer()

# Summarize a document
results = rag.summarize_document("your_document.pdf")
rag.display_results(results)
```

### News Processing Only
```python
from news_retriever import NewsRetriever

# Fetch latest news
retriever = NewsRetriever()
articles = retriever.fetch_all_news(articles_per_source=3)
```

### Custom News Processing
```python
# Process with custom settings
summary_data = rag.process_all_news(articles_per_source=5)
rag.display_news_summaries(summary_data)
rag.save_summaries_to_markdown(summary_data)
```

## 🔄 How It Works

### Document Processing Pipeline
1. **Document Loading**: Multi-format parsing (PDF, TXT, MD, JSON)
2. **Text Preprocessing**: Cleaning and normalization
3. **Smart Chunking**: Sentence-aware splitting with overlap
4. **Vector Database**: Semantic embeddings with ChromaDB
5. **Retrieval**: Context-aware chunk selection
6. **Generation**: BART-powered summarization

### News Processing Pipeline
1. **News Fetching**: RSS feed parsing from multiple sources
2. **Content Extraction**: Full article text retrieval
3. **RAG Processing**: Same pipeline as documents
4. **Batch Summarization**: Automated processing of multiple articles
5. **Multi-format Output**: JSON, Markdown, and console display

## 📊 Output Examples

### Console Output
```
==================================================
LATEST NEWS SUMMARIES
==================================================
Generated at: 2025-06-17T14:30:25
Total articles: 15

1. Global Climate Summit Reaches Historic Agreement
   Source: BBC World
   Summary: World leaders agree on new emissions targets...
   Compression: 450 → 85 words
--------------------------------------------------
```

### File Outputs
- **Raw Data**: `news_data/raw_news_20250617_143025.json`
- **Summaries**: `news_data/news_summaries_20250617_143025.json`
- **Markdown**: `news_data/news_summaries_20250617_143025.md`

## 🛠️ Models & Technology

### AI Models
- **Embeddings**: `all-MiniLM-L6-v2` (SentenceTransformers)
- **Summarization**: `facebook/bart-large-cnn`
- **Vector Database**: ChromaDB

### News Sources
- BBC World News
- Reuters World News
- CNN International
- Associated Press
- Al Jazeera English
- NPR World News
- The Guardian World

## 📁 Project Structure

```
├── main.py                    # Main execution script
├── news_retriever.py          # News fetching module
├── rag_summarizer.py         # RAG implementation
├── requirements.txt          # Python dependencies
├── README.md                # This documentation
├── news_data/               # Generated news files
│   ├── raw_news_*.json     # Original articles
│   ├── news_summaries_*.json # Structured summaries
│   └── news_summaries_*.md   # Human-readable format
└── sample_docs/             # Test documents (auto-generated)
    ├── ai_article.txt
    ├── climate_report.txt
    └── tech_trends.txt
```

## ⚙️ Configuration

### Customizable Parameters
```python
# News fetching
articles_per_source = 3      # Articles per news source
max_articles_total = 20      # Total article limit

# Text processing
chunk_size = 300            # Words per chunk
overlap = 50               # Overlap between chunks

# Summarization
max_summary_length = 150    # Maximum summary length
min_summary_length = 50     # Minimum summary length
top_k_chunks = 5           # Chunks for summarization
```

### Adding News Sources
```python
# In news_retriever.py
self.news_sources = {
    'Your Source': 'https://example.com/rss.xml',
    # Add more RSS feeds here
}
```

## 🔧 Advanced Features

### Batch Processing
- Process multiple documents simultaneously
- Automated news updates on schedule
- Bulk summarization with progress tracking

### Quality Control
- Similarity scoring for chunk relevance
- Compression ratio monitoring
- Error handling with fallback mechanisms

### Export Options
- JSON for structured data
- Markdown for human reading
- CSV for data analysis
- Custom formats via templates


## 🚨 Notes & Limitations

### Performance
- **CPU optimized** (GPU acceleration available)
- **Memory efficient** chunking for large documents
- **Rate limiting** for respectful news scraping

### Content Guidelines
- News sources may have access restrictions
- Content extraction depends on website structure
- Some paywalled content may not be accessible

### Error Handling
- Graceful degradation on network issues
- Fallback summarization methods
- Comprehensive logging for debugging

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues
- **NLTK data missing**: Run `nltk.download('punkt')`
- **ChromaDB errors**: Clear existing collections
- **Network timeouts**: Check internet connection
- **Memory issues**: Reduce `chunk_size` or `articles_per_source`

### Getting Help
- Check console output for detailed error messages
- Verify all dependencies are installed
- Ensure internet connection for news fetching
- Review configuration parameters

---

*Keep yourself updated with the world's latest news through AI-powered summarization!* 🌍📰
