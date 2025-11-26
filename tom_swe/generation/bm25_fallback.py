"""
Pure Python BM25 implementation fallback when bm25s is not available.

This is a simplified, pure-Python implementation that:
1. Limits search to latest 10 conversations (performance optimization)
2. Uses basic Porter stemming approximation
3. Includes English stopword filtering
4. Implements standard BM25 ranking algorithm

Install the full version with: pip install tom-swe[search]
"""

import re
import math
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter


# English stopwords (common set - 179 words)
STOPWORDS: Set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


class SimpleStemmer:
    """
    Simple stemming approximation without PyStemmer.

    Uses basic suffix stripping rules inspired by Porter stemmer.
    Not as accurate as PyStemmer but sufficient for basic BM25.
    """

    # Common suffix patterns (ordered by priority - longer first)
    SUFFIXES = [
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ational", "ate"),
        ("tional", "tion"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
        ("ousli", "ous"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ing", ""),
        ("ed", ""),
        ("ies", "i"),
        ("s", ""),
    ]

    def stem(self, word: str) -> str:
        """Apply simple stemming rules to a word."""
        if len(word) <= 3:  # Don't stem very short words
            return word

        word = word.lower()

        # Try each suffix pattern
        for suffix, replacement in self.SUFFIXES:
            if word.endswith(suffix):
                # Keep stem if it's at least 2 characters
                stem = word[: -len(suffix)] + replacement
                if len(stem) >= 2:
                    return stem

        return word


def tokenize(
    text: str, remove_stopwords: bool = True, stemmer: Optional[SimpleStemmer] = None
) -> List[str]:
    """
    Tokenize text into words with optional stopword removal and stemming.

    Args:
        text: Text to tokenize
        remove_stopwords: Whether to remove stopwords
        stemmer: Stemmer instance to use

    Returns:
        List of tokens
    """
    if stemmer is None:
        stemmer = SimpleStemmer()

    # Convert to lowercase and extract words (alphanumeric + underscores)
    tokens = re.findall(r"\b\w+\b", text.lower())

    # Remove stopwords if requested
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    # Apply stemming
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


class BM25:
    """
    Pure Python BM25 implementation.

    Implements the standard BM25 ranking formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))

    Where:
    - f(qi,D) = frequency of query term qi in document D
    - |D| = length of document D
    - avgdl = average document length
    - k1 = term frequency saturation parameter (default 1.5)
    - b = length normalization parameter (default 0.75)
    - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 ranker.

        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (typically 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus_tokens: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0  # Number of documents

    def index(self, corpus_tokens: List[List[str]]) -> None:
        """
        Build BM25 index from tokenized corpus.

        Args:
            corpus_tokens: List of tokenized documents
        """
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)

        if self.N == 0:
            self.avgdl = 0.0
            return

        # Calculate document lengths
        self.doc_lengths = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)

        # Calculate document frequencies
        self.doc_freqs = {}
        for tokens in corpus_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # Calculate IDF scores
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            # Standard BM25 IDF formula
            idf_score = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[token] = idf_score

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """
        Calculate BM25 scores for all documents given a query.

        Args:
            query_tokens: Tokenized query

        Returns:
            List of BM25 scores for each document
        """
        if not query_tokens:
            return [0.0] * self.N

        scores = [0.0] * self.N

        for doc_idx, doc_tokens in enumerate(self.corpus_tokens):
            # Count term frequencies in document
            term_freqs = Counter(doc_tokens)
            doc_len = self.doc_lengths[doc_idx]

            # Calculate BM25 score for this document
            score = 0.0
            for query_token in query_tokens:
                if query_token not in self.idf:
                    continue

                # Get term frequency in document
                tf = term_freqs.get(query_token, 0)

                # BM25 formula components
                idf_score = self.idf[query_token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )

                score += idf_score * (numerator / denominator)

            scores[doc_idx] = score

        return scores

    def retrieve(
        self, query_tokens: List[str], k: int = 10
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Retrieve top-k documents for a query.

        Args:
            query_tokens: Tokenized query
            k: Number of top results to return

        Returns:
            Tuple of (doc_indices, scores) where each is a 2D list to match bm25s API
            - doc_indices: [[idx1, idx2, ...]] - top k document indices
            - scores: [[score1, score2, ...]] - corresponding scores
        """
        # Get scores for all documents
        scores = self.get_scores(query_tokens)

        # Sort by score (descending) and get top k
        scored_docs = [(idx, score) for idx, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Take top k
        top_k = scored_docs[:k]

        # Format to match bm25s return format: 2D arrays with shape (1, k)
        doc_indices = [[idx for idx, _ in top_k]]
        doc_scores = [[score for _, score in top_k]]

        return doc_indices, doc_scores


def tokenize_corpus(
    corpus: List[str], stopwords: str = "en", stemmer: Optional[SimpleStemmer] = None
) -> List[List[str]]:
    """
    Tokenize a corpus of documents.

    Args:
        corpus: List of document strings
        stopwords: "en" for English stopwords, or None for no filtering
        stemmer: Stemmer instance (uses SimpleStemmer if None)

    Returns:
        List of tokenized documents
    """
    if stemmer is None:
        stemmer = SimpleStemmer()

    remove_stops = stopwords == "en"
    return [
        tokenize(doc, remove_stopwords=remove_stops, stemmer=stemmer) for doc in corpus
    ]
