# utils.py

import re
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text: str) -> str:
    """
    Basic text cleaner for strategy rules.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9%<>.= ]+", "", text)
    return text.strip()

def tokenize_rules(rules: List[str]) -> List[str]:
    """
    Cleans and tokenizes each strategy rule.
    """
    return [clean_text(rule) for rule in rules]

def compute_tfidf_similarity(set1: List[str], set2: List[str]) -> float:
    """
    Optional: Compare two rule sets using TF-IDF cosine similarity.
    """
    corpus = [" ".join(set1), " ".join(set2)]
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(corpus)
    sim_matrix = cosine_similarity(vectors)
    return sim_matrix[0][1]

def score_strategy_length(rules: List[str]) -> float:
    """
    Simple scoring logic based on number of rules.
    (Can be expanded for quality scoring.)
    """
    return min(1.0, len(rules) / 5.0)  # Max score if 5 rules




