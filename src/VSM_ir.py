import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)

# Define a set named indonesian_stopwords
indonesian_stopwords = set(stopwords.words('indonesian'))

# Initialize PorterStemmer
ps = PorterStemmer()

# Helper preprocessing functions
def case_folding(text):
    return text.lower()

def normalize_text(text):
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)  # Double escape for string literal
    text = re.sub(r'[^\w\s]', '', text) # Double escape for string literal
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [word for word in tokens if word not in indonesian_stopwords]

def stem_tokens(tokens):
    return [ps.stem(word) for word in tokens]

# Main preprocessing function
def preprocess_query(query):
    query = case_folding(query)
    query = normalize_text(query)
    tokens = tokenize_text(query)
    tokens_no_stopwords = remove_stopwords(tokens)
    stemmed_tokens = stem_tokens(tokens_no_stopwords)
    return ' '.join(stemmed_tokens) # Join back to a string for TfidfVectorizer

# VSM Core Functions
def compute_tfidf_sparse(docs, sublinear_tf=False, vocabulary=None):
    """Calculates TF-IDF matrix for a list of documents."""
    tfidf_vectorizer = TfidfVectorizer(
        sublinear_tf=sublinear_tf,
        use_idf=True,
        stop_words='english', # NLTK stopwords were used in preprocessing, but this ensures consistency for vectorizer
        smooth_idf=True,
        vocabulary=vocabulary # Use a pre-defined vocabulary if provided
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    return tfidf_matrix, tfidf_vectorizer

def vectorize_query(query_string, tfidf_vectorizer):
    """Vectorizes a query string using a pre-fitted TfidfVectorizer."""
    processed_query_text = preprocess_query(query_string)
    query_tfidf_vector = tfidf_vectorizer.transform([processed_query_text])
    return query_tfidf_vector

def calculate_cosine_similarity(query_vector, document_matrix):
    """Calculates cosine similarity between a query vector and a document matrix."""
    similarities = cosine_similarity(query_vector, document_matrix)
    return similarities.flatten()

def retrieve_top_k_documents(query_string, document_matrix, tfidf_vectorizer, k=5, processed_docs=None):
    """Retrieves the top-k documents most similar to the query using TF-IDF and cosine similarity."""
    query_vector = vectorize_query(query_string, tfidf_vectorizer)
    similarities = calculate_cosine_similarity(query_vector, document_matrix)

    top_k_indices = np.argsort(similarities)[::-1]

    results = []
    valid_results_count = 0
    for doc_idx in top_k_indices:
        score = similarities[doc_idx]
        if score > 0 and valid_results_count < k:
            doc_id = doc_idx + 1
            snippet = ""
            if processed_docs and doc_idx < len(processed_docs):
                snippet = processed_docs[doc_idx][:150] + "..." # Display first 150 chars
            results.append({"doc_id": doc_id, "score": score, "snippet": snippet})
            valid_results_count += 1
        elif valid_results_count >= k:
            break

    return results
