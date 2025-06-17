from rag_summarizer import RAGSummarizer

if __name__ == "__main__":
    print("Enhanced RAG News Summarizer")
    print("=" * 50)

    rag_summarizer = RAGSummarizer()
    summary_data = rag_summarizer.process_all_news(articles_per_source=2)

    if summary_data:
        rag_summarizer.display_news_summaries(summary_data)
        markdown_file = rag_summarizer.save_summaries_to_markdown(summary_data)
        
    else:
        print("Failed to fetch news articles. Please check your internet connection.")
