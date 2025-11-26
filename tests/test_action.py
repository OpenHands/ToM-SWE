"""
Simple tests for the action module.
"""

import pytest
from unittest.mock import Mock, patch
from tom_swe.generation.action import ActionExecutor
from tom_swe.generation.dataclass import SearchFileParams, ReadFileParams


def test_search_action_basic() -> None:
    """Test basic search action functionality."""
    # Create mock file store
    mock_file_store = Mock()
    mock_file_store.list.return_value = ["test_file.json"]
    mock_file_store.read.return_value = '{"content": "test content"}'

    # Create action executor
    executor = ActionExecutor(user_id="test_user", file_store=mock_file_store)

    # Create search parameters
    params = SearchFileParams(
        query="test", search_scope="session_analyses", search_method="string_match"
    )

    # Execute search
    result = executor._string_search(params)

    # Basic assertions
    assert "Found 1 files" in result
    assert "test_file.json" in result


def test_search_action_no_results() -> None:
    """Test search when no results found."""
    # Create mock file store
    mock_file_store = Mock()
    mock_file_store.list.return_value = ["test_file.json"]
    mock_file_store.read.return_value = '{"content": "different content"}'

    # Create action executor
    executor = ActionExecutor(user_id="test_user", file_store=mock_file_store)

    # Create search parameters
    params = SearchFileParams(
        query="nonexistent",
        search_scope="session_analyses",
        search_method="string_match",
    )

    # Execute search
    result = executor._string_search(params)

    # Basic assertions
    assert "No files found" in result


def test_bm25_search() -> None:
    """Test BM25 search functionality."""
    # Create mock file store with multiple documents
    mock_file_store = Mock()
    mock_file_store.list.return_value = ["file1.json", "file2.json", "file3.json"]
    mock_file_store.read.side_effect = [
        '{"content": "python programming guide"}',
        '{"content": "machine learning algorithms"}',
        '{"content": "data structures and algorithms"}',
    ]

    # Create action executor
    executor = ActionExecutor(user_id="test_user", file_store=mock_file_store)

    # Create search parameters
    params = SearchFileParams(
        query="python programming",
        search_scope="session_analyses",
        search_method="bm25",
        max_results=3,
    )

    # Execute BM25 search
    result = executor._action_search_file(params)

    # Debug: print the actual result to see the format
    print(f"\nBM25 Search Result:\n{result}")

    # Basic assertions
    assert "Found 3 files (BM25 ranked)" in result
    assert "file1.json" in result

    # Verify that file1.json is the first (highest ranked) result
    # The result format is: [Score: X.XX] filename: content
    lines = result.split("\n")
    # Find the first line that contains a filename
    first_file_line = None
    for line in lines:
        if "]" in line and ".json:" in line:
            first_file_line = line
            break

    assert first_file_line is not None, "No file line found in result"
    assert (
        "file1.json" in first_file_line
    ), f"file1.json should be first, but first line is: {first_file_line}"


def test_read_action_basic() -> None:
    """Test basic read action functionality."""
    # Create mock file store
    mock_file_store = Mock()
    test_content = "This is a test file content with some data that we want to read."
    mock_file_store.read.return_value = test_content

    # Create action executor
    executor = ActionExecutor(user_id="test_user", file_store=mock_file_store)

    # Create read parameters with custom character range
    params = ReadFileParams(
        file_path="test_file.json", character_start=0, character_end=25
    )

    # Execute read action
    result = executor._action_read_file(params)

    # Verify file store read was called correctly
    mock_file_store.read.assert_called_once_with("test_file.json")

    # Verify the correct character range was extracted
    expected_result = test_content[0:25]  # "This is a test file cont"
    assert result == expected_result


def test_pure_python_bm25_directly() -> None:
    """Test pure Python BM25 implementation directly."""
    from tom_swe.generation.bm25_fallback import (
        BM25,
        SimpleStemmer,
        tokenize,
        tokenize_corpus,
    )

    # Test tokenization with stopword removal
    text = "The quick brown fox jumps over the lazy dog"
    tokens = tokenize(text, remove_stopwords=True)
    assert "the" not in tokens  # Stopword removed
    assert len(tokens) > 0

    # Test stemming
    stemmer = SimpleStemmer()
    # Just verify stemming changes the word
    assert stemmer.stem("running") != "running"
    assert len(stemmer.stem("running")) <= len("running")

    # Test BM25 ranking
    corpus = [
        "python programming language",
        "machine learning with python",
        "data science and python",
    ]

    stemmer = SimpleStemmer()
    corpus_tokens = tokenize_corpus(corpus, stopwords="en", stemmer=stemmer)

    retriever = BM25()
    retriever.index(corpus_tokens)

    query = "python programming"
    query_tokens = tokenize(query, remove_stopwords=True, stemmer=stemmer)
    doc_indices, scores = retriever.retrieve(query_tokens, k=3)

    # Verify format matches bm25s API
    assert len(doc_indices) == 1  # 2D array
    assert len(doc_indices[0]) == 3  # k results
    assert len(scores) == 1
    assert len(scores[0]) == 3

    # First result should be most relevant (doc 0: "python programming language")
    assert doc_indices[0][0] == 0
    assert scores[0][0] > 0  # Should have positive score


def test_conversation_limit_with_fallback() -> None:
    """Test that pure Python BM25 limits to 10 conversations."""
    with patch("tom_swe.generation.action.HAS_BM25S", False):
        mock_file_store = Mock()

        # Create 20 mock files
        files = [f"file{i}.json" for i in range(20)]
        mock_file_store.list.return_value = files
        mock_file_store.read.return_value = (
            '{"content": "test content", "last_updated": "2025-01-01"}'
        )

        executor = ActionExecutor(user_id="test_user", file_store=mock_file_store)

        # This should limit to 10 conversations internally
        result = executor._get_content_by_scope(
            "session_analyses",
            latest_first=True,
            limit=50,  # Request 50 but should get 10
        )

        # Should return at most 10 results
        assert len(result) <= 10


def test_bm25_fallback_consistency() -> None:
    """
    Test that pure Python BM25 produces reasonable results.

    Note: Results won't be identical to bm25s due to different stemming,
    but ranking should be similar for simple queries.
    """
    corpus_data = [
        '{"content": "implementing authentication system"}',
        '{"content": "user authentication and authorization"}',
        '{"content": "database schema design"}',
    ]

    mock_file_store = Mock()
    mock_file_store.list.return_value = ["f1.json", "f2.json", "f3.json"]
    mock_file_store.read.side_effect = corpus_data

    executor = ActionExecutor(user_id="test_user", file_store=mock_file_store)

    params = SearchFileParams(
        query="authentication",
        search_scope="session_analyses",
        search_method="bm25",
        max_results=2,
    )

    result = executor._action_search_file(params)

    # Both implementations should:
    # 1. Return results
    assert "Found" in result

    # 2. Include authentication-related files
    assert "f1.json" in result or "f2.json" in result

    # 3. Not fail or throw errors
    assert "Error" not in result


if __name__ == "__main__":
    pytest.main([__file__])
