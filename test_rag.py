#!/usr/bin/env python3
"""
Test script for the new RAG implementation with TF-IDF vectorization.
"""

import sys
sys.path.insert(0, 'src')

from tagent.semantic_search import TFIDFRagContextManager, RagDocument
from tagent.models import MemoryItem, EnhancedAgentState
from tagent.memory_manager import EnhancedContextManager


def test_basic_rag_functionality():
    """Test basic RAG functionality."""
    print("=== Testing Basic RAG Functionality ===\n")
    
    # Create RAG context manager
    rag_manager = TFIDFRagContextManager(goal="Test semantic search capabilities")
    
    # Add some test memories
    test_memories = [
        MemoryItem(
            content="Successfully executed web search for Python tutorials",
            type="execution_success",
            relevance="web_search python tutorials"
        ),
        MemoryItem(
            content="Failed to connect to database due to timeout",
            type="execution_failure",
            relevance="database connection timeout"
        ),
        MemoryItem(
            content="Learned that TF-IDF is effective for document similarity",
            type="fact",
            relevance="tfidf similarity search"
        ),
        MemoryItem(
            content="Planning strategy: break complex tasks into smaller subtasks",
            type="strategy",
            relevance="planning task decomposition"
        )
    ]
    
    # Store memories
    rag_manager.store_memories(test_memories, "test_context")
    
    # Test semantic search
    print("1. Testing semantic search...")
    results = rag_manager.semantic_search("search for python programming", top_k=3)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"   Result {i}: [score: {score:.3f}] {doc.content[:50]}...")
    
    print()
    
    # Test search with different query
    print("2. Testing search for database-related content...")
    results = rag_manager.semantic_search("database connection issues", top_k=2)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"   Result {i}: [score: {score:.3f}] {doc.content[:50]}...")
    
    print()
    
    # Test search for planning content
    print("3. Testing search for planning strategies...")
    results = rag_manager.semantic_search("how to plan complex tasks", top_k=2)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"   Result {i}: [score: {score:.3f}] {doc.content[:50]}...")
    
    print()
    
    # Test memory statistics
    print("4. Memory statistics:")
    stats = rag_manager.get_memory_stats()
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Memory documents: {stats['memory_documents']}")
    print(f"   Vectorizer fitted: {stats['vectorizer_fitted']}")
    print(f"   Vocabulary size: {stats['vocabulary_size']}")
    
    print()


def test_enhanced_context_manager():
    """Test the enhanced context manager with state-based retrieval."""
    print("=== Testing Enhanced Context Manager ===\n")
    
    # Create enhanced context manager
    context_manager = EnhancedContextManager(goal="Build a web scraper for news articles")
    
    # Add some domain-specific memories
    domain_memories = [
        MemoryItem(
            content="Web scraping requires proper handling of HTTP headers and rate limiting",
            type="fact",
            relevance="web_scraping http headers rate_limiting"
        ),
        MemoryItem(
            content="BeautifulSoup is effective for parsing HTML content",
            type="strategy",
            relevance="html_parsing beautifulsoup"
        ),
        MemoryItem(
            content="Failed to scrape due to anti-bot protection",
            type="execution_failure",
            relevance="scraping anti_bot protection"
        ),
        MemoryItem(
            content="Successfully extracted article titles and content",
            type="execution_success",
            relevance="extraction articles content"
        )
    ]
    
    context_manager.store_memories(domain_memories, "web_scraping")
    
    # Create a mock state for testing
    mock_state = EnhancedAgentState(
        goal="Build a web scraper for news articles",
        current_phase="plan",
        available_tools=["web_request", "html_parser", "text_extractor"],
        collected_data={},
        failure_reason=None,
        context_history=[]
    )
    
    # Test context retrieval for planning
    print("1. Testing context for planning phase...")
    context = context_manager.get_context_for_current_state(mock_state)
    from src.tagent.token_utils import format_token_size_info
    print(f"   Context: {format_token_size_info(context)}")
    print(f"   Context preview: {context[:200]}...")
    print()
    
    # Test context retrieval for execution
    print("2. Testing context for execution phase...")
    mock_state.current_phase = "execute"
    context = context_manager.get_context_for_current_state(mock_state)
    from src.tagent.token_utils import format_token_size_info
    print(f"   Context: {format_token_size_info(context)}")
    print(f"   Context preview: {context[:200]}...")
    print()
    
    # Test with failure context
    print("3. Testing context with failure...")
    mock_state.failure_reason = "HTTP 403 error - blocked by anti-bot system"
    mock_state.last_result = "failed"
    mock_state.last_action = "web_request"
    context = context_manager.get_context_for_current_state(mock_state)
    from src.tagent.token_utils import format_token_size_info
    print(f"   Context: {format_token_size_info(context)}")
    print(f"   Context preview: {context[:200]}...")
    print()
    
    # Test memory summary
    print("4. Memory summary for finalization:")
    summary = context_manager.get_memory_summary_for_finalize()
    print(f"   Total memories: {summary['total_memories']}")
    print(f"   Memory types: {summary['memory_types']}")
    print(f"   Semantic search enabled: {summary['semantic_search_enabled']}")
    print(f"   Vectorizer features: {len(summary['vectorizer_features'])} features")
    
    print()


def test_document_similarity():
    """Test document similarity functionality."""
    print("=== Testing Document Similarity ===\n")
    
    rag_manager = TFIDFRagContextManager(goal="Test similarity search")
    
    # Add test documents
    test_docs = [
        RagDocument(
            id="doc1",
            content="Python is a programming language that is easy to learn and powerful",
            doc_type="fact",
            keywords=["python", "programming", "language"]
        ),
        RagDocument(
            id="doc2", 
            content="Machine learning with Python uses libraries like scikit-learn and TensorFlow",
            doc_type="fact",
            keywords=["machine", "learning", "python", "scikit-learn"]
        ),
        RagDocument(
            id="doc3",
            content="Web development with Flask and Django frameworks in Python",
            doc_type="fact",
            keywords=["web", "development", "flask", "django", "python"]
        )
    ]
    
    # Add documents manually
    for doc in test_docs:
        rag_manager.documents[doc.id] = doc
    
    # Update embeddings
    rag_manager._update_embeddings()
    
    # Test similarity search
    print("1. Search for 'programming languages':")
    results = rag_manager.semantic_search("programming languages", top_k=2)
    for i, (doc, score) in enumerate(results, 1):
        print(f"   Result {i}: [score: {score:.3f}] {doc.content}")
    
    print()
    
    print("2. Search for 'web frameworks':")
    results = rag_manager.semantic_search("web frameworks", top_k=2) 
    for i, (doc, score) in enumerate(results, 1):
        print(f"   Result {i}: [score: {score:.3f}] {doc.content}")
    
    print()
    
    print("3. Search for 'machine learning':")
    results = rag_manager.semantic_search("machine learning", top_k=2)
    for i, (doc, score) in enumerate(results, 1):
        print(f"   Result {i}: [score: {score:.3f}] {doc.content}")
    
    print()


def main():
    """Run all tests."""
    print("Starting RAG Implementation Tests...\n")
    
    try:
        test_basic_rag_functionality()
        test_enhanced_context_manager()
        test_document_similarity()
        
        print("=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()