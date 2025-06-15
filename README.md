# RAG Document Summarizer

A basic implementation of Retrieval-Augmented Generation (RAG) for document summarization using semantic chunking and vector search.

## Features

- Document parsing (PDF, TXT, MD)
- Semantic text chunking with overlap
- Vector embeddings using SentenceTransformers
- Semantic retrieval with ChromaDB
- Summary generation with BART model

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the code:
```python
python rag_summarizer.py
```

## How it works

1. **Document Loading**: Loads PDF or TXT files
2. **Text Chunking**: Splits document into overlapping chunks
3. **Vector Database**: Creates embeddings and stores in ChromaDB
4. **Retrieval**: Finds most relevant chunks for summarization
5. **Generation**: Uses BART model to generate final summary

## Sample Usage

```python
# Initialize
rag = SimpleRAGSummarizer()

# Summarize a document
results = rag.summarize_document("your_document.pdf")
rag.display_results(results)
```

## Output

The system shows:
- Original document statistics
- Retrieved chunk previews with similarity scores
- Final generated summary

## Models Used

- **Embeddings**: all-MiniLM-L6-v2 (SentenceTransformers)
- **Summarization**: facebook/bart-large-cnn
- **Vector DB**: ChromaDB

## Files Structure

```
├── document_loader.py      # Utils code
├── rag_summarizer.py      # Main code
├── requirements.txt       # Dependencies
├── README.md             # This file
└── sample_docs/          # Auto-generated test documents
    ├── ai_article.txt
    ├── climate_report.txt
    └── tech_trends.txt
```

## Notes

- Designed for Kaggle notebooks
- Uses CPU by default (GPU if available)
- Simple error handling for learning purposes
- Creates sample documents automatically for testing
