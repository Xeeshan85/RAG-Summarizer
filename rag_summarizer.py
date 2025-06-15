from document_loader import DocumentLoader
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
import json

import PyPDF2
import markdown
from typing import List, Dict, Tuple, Optional

from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt', quiet=True)

# For embeddings and vector search
from sentence_transformers import SentenceTransformer
import chromadb
# For LLM (using transformers)
from transformers import pipeline
import torch


import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class RAGSummarizer:
    def __init__(self):
        ## Load sentence transformer for embeddings
        self.document_loader = DocumentLoader()
        self.supported_formats = self.document_loader.supported_formats
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        ## Setup ChromaDB
        self.client = chromadb.Client()
        self.collection = None
        
        ## Load summarization model
        self.summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        print("Models loaded successfully!")
    
    def load_document(self, path: str) -> str:
        return self.document_loader.load_document(path)
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text) ## remove extra spaces
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text) ## remove special chars, but punctuations 
        return text.strip()
    
    def chunk_text(self, text, chunk_size=300, overlap=50):
        text = self._clean_text(text)
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) > chunk_size: ## - If too long save current
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    ## Start new chunk with some overlap
                    words = current_chunk.split()
                    if len(words) > overlap:
                        current_chunk = " ".join(words[-overlap:]) + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        ## Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_vector_db(self, chunks):
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        ## Delete existing collection if it exists
        try:
            self.client.delete_collection("docs")
        except:
            pass
        
        ## Create new collection
        self.collection = self.client.create_collection("docs")
        
        ## Add chunks to collection
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=chunk_ids)
        print("Vector database created!")
    
    def retrieve_relevant_chunks(self, query, top_k=5):
        ## Get most relevant chunks for query
        if self.collection is None:
            raise ValueError("No vector database created!")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results['documents'][0], results['distances'][0]
    
    def generate_summary(self, chunks, max_length=150):
        ## Combine chunks
        combined_text = " ".join(chunks)
        
        ## input length limiit to avoid model limits
        if len(combined_text.split()) > 1000:
            combined_text = " ".join(combined_text.split()[:1000])
        
        try:
            ## Generate summary
            summary = self.summarizer(
                combined_text,
                max_length=max_length,
                min_length=50,
                do_sample=False
            )[0]['summary_text']
            
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            ## Fallback to simple extractive summary
            sentences = sent_tokenize(combined_text)
            return " ".join(sentences[:3])  ## Just take first 3 sentences
    
    def summarize_document(self, file_path, summary_query="Summarize"):
        print(f"\nProcessing: {file_path}")
        
        ## Load document
        text = self.load_document(file_path)
        print(f"Document loaded: {len(text.split())} words")
        
        ## Chunk text
        chunks = self.chunk_text(text)
        print(f"Created {len(chunks)} chunks")
        
        ## Create vector database
        self.create_vector_db(chunks)
        
        ## Retrieve relevant chunks
        relevant_chunks, distances = self.retrieve_relevant_chunks(summary_query)
        print(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        ## Generate summary
        summary = self.generate_summary(relevant_chunks)
        
        ## Results
        results = {
            'file_path': file_path,
            'original_length': len(text.split()),
            'num_chunks': len(chunks),
            'retrieved_chunks': relevant_chunks,
            'similarity_scores': [1 - d for d in distances],  ## Convert distance to similarity
            'summary': summary
        }
        
        return results
    
    def display_results(self, results):
        print("\n" + "="*60)
        print("SUMMARIZATION RESULTS")
        print("="*60)
        
        print(f"Document: {os.path.basename(results['file_path'])}")
        print(f"Original length: {results['original_length']} words")
        print(f"Number of chunks: {results['num_chunks']}")
        print(f"Summary length: {len(results['summary'].split())} words")
        
        print(f"\nRetrieved Chunks (Top 3):")
        for i, (chunk, score) in enumerate(zip(results['retrieved_chunks'][:3], 
                                              results['similarity_scores'][:3])):
            print(f"\n{i+1}. Similarity: {score:.3f}")
            print(f"   Text: {chunk[:200]}...")
        
        print(f"\nFINAL SUMMARY:")
        print("-" * 40)
        print(results['summary'])
        print("-" * 40)


# Sample documents for testing
def create_sample_docs():
    os.makedirs("sample_docs", exist_ok=True)
    
    # Sample 1: AI article
    ai_doc = """
    Artificial Intelligence: The Future is Now
    
    Artificial Intelligence (AI) has become one of the most transformative technologies of our time. 
    From healthcare to finance, AI is revolutionizing industries and changing how we live and work.
    
    Machine learning, a subset of AI, enables computers to learn from data without explicit programming.
    Deep learning, using neural networks, has achieved breakthroughs in image recognition, natural language processing, and game playing.
    
    Applications of AI are everywhere. In healthcare, AI helps diagnose diseases and discover new drugs.
    In finance, it detects fraud and enables algorithmic trading. Self-driving cars use AI for navigation and safety.
    
    However, AI also raises important ethical questions about privacy, bias, and job displacement.
    As AI continues to advance, we must ensure it benefits all of humanity while addressing these challenges.
    
    The future of AI looks promising with emerging technologies like quantum computing and brain-computer interfaces.
    These developments could lead to even more powerful AI systems that can solve complex global problems.
    """
    
    with open("sample_docs/ai_article.txt", "w") as f:
        f.write(ai_doc)
    
    # Sample 2: Climate change report
    climate_doc = """
    Climate Change: Understanding the Global Challenge
    
    Climate change represents one of the most pressing challenges of our generation.
    Rising global temperatures, melting ice caps, and increasing frequency of extreme weather events
    are clear indicators that our planet's climate system is undergoing significant changes.
    
    The primary cause of current climate change is human activities, particularly the emission of greenhouse gases
    from burning fossil fuels. Carbon dioxide levels have increased by over 40% since pre-industrial times.
    
    The impacts of climate change are already visible worldwide. Sea levels are rising, threatening coastal communities.
    Droughts and floods are becoming more frequent and severe. Arctic ice is melting at an unprecedented rate.
    
    However, there is still hope. Renewable energy technologies like solar and wind power are becoming cheaper and more efficient.
    Many countries are committing to net-zero emissions targets. Innovation in clean technology is accelerating.
    
    Individual actions also matter. Reducing energy consumption, using public transportation, and supporting
    sustainable practices can contribute to the solution. Education and awareness are key to driving change.
    
    The transition to a sustainable future requires global cooperation and immediate action.
    The choices we make today will determine the planet we leave for future generations.
    """
    
    with open("sample_docs/climate_report.txt", "w") as f:
        f.write(climate_doc)
    
    # Sample 3: Technology trends
    tech_doc = """
    Technology Trends Shaping 2024
    
    The technology landscape continues to evolve rapidly, with several key trends emerging in 2024.
    These developments are reshaping industries and creating new opportunities for innovation.
    
    Cloud computing has become the backbone of modern business operations. Companies are migrating
    to cloud-first strategies, enabling scalability, flexibility, and cost savings.
    
    Cybersecurity has gained critical importance as digital threats become more sophisticated.
    Zero-trust security models and AI-powered threat detection are becoming standard practices.
    
    The Internet of Things (IoT) is connecting billions of devices, creating smart cities and homes.
    Edge computing is bringing processing power closer to data sources, reducing latency and improving performance.
    
    Blockchain technology is finding applications beyond cryptocurrency, including supply chain management,
    digital identity, and smart contracts. Web3 and decentralized applications are gaining traction.
    
    Quantum computing, while still in early stages, promises to solve complex problems that are impossible
    for classical computers. Major tech companies are investing heavily in quantum research.
    
    The future of technology will be defined by the convergence of these trends, creating new possibilities
    for solving global challenges and improving human life.
    """
    
    with open("sample_docs/tech_trends.txt", "w") as f:
        f.write(tech_doc)
    
    return ["sample_docs/ai_article.txt", "sample_docs/climate_report.txt", "sample_docs/tech_trends.txt"]


if __name__ == "__main__":
    print("Simple RAG Document Summarizer")
    print("=" * 40)
    
    rag_summarizer = RAGSummarizer()
    sample_files = create_sample_docs()
    
    ## Process each document
    for file_path in sample_files:
        try:
            results = rag_summarizer.summarize_document(file_path)
            rag_summarizer.display_results(results)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("Done! Check the sample_docs folder for the test documents.")
