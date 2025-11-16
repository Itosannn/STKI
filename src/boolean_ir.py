import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

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
    return stemmed_tokens

# Functions to build IR components
def build_vocabulary(processed_documents):
    vocabulary = set()
    for doc in processed_documents:
        for term in doc.split(): # Assuming docs are strings of space-separated terms
            vocabulary.add(term)
    return sorted(list(vocabulary))

def build_incidence_matrix(processed_documents, vocabulary):
    term_to_idx = {term: i for i, term in enumerate(vocabulary)}
    # Initialize with 0s
    incidence_matrix = [[0] * len(vocabulary) for _ in processed_documents]

    for i, doc in enumerate(processed_documents):
        for term in doc.split():
            if term in term_to_idx:
                incidence_matrix[i][term_to_idx[term]] = 1
    return incidence_matrix

def build_inverted_index(processed_documents, vocabulary):
    inverted_index = defaultdict(list)
    for i, doc_text in enumerate(processed_documents):
        doc_id = i + 1  # 1-indexed document IDs
        unique_terms_in_doc = set(doc_text.split()) # Get unique terms per document
        for term in unique_terms_in_doc:
            inverted_index[term].append(doc_id)
    return dict(inverted_index)

# Boolean Query Parsing
def parse_boolean_query_string(query_string):
    # Normalize operators to ensure consistent splitting, convert to upper for matching
    query_string_upper = query_string.upper()
    
    # Check for explicit binary operators
    if ' AND ' in query_string_upper:
        parts = query_string.split(' AND ')
        if len(parts) == 2:
            term1_processed = preprocess_query(parts[0])
            term2_processed = preprocess_query(parts[1])
            return {"terms": [term1_processed, term2_processed], "operator": "AND"}
    elif ' OR ' in query_string_upper:
        parts = query_string.split(' OR ')
        if len(parts) == 2:
            term1_processed = preprocess_query(parts[0])
            term2_processed = preprocess_query(parts[1])
            return {"terms": [term1_processed, term2_processed], "operator": "OR"}
    elif ' NOT ' in query_string_upper:
        parts = query_string.split(' NOT ')
        if len(parts) == 2:
            term1_processed = preprocess_query(parts[0])
            term2_processed = preprocess_query(parts[1])
            return {"terms": [term1_processed, term2_processed], "operator": "NOT"}
    
    # If no explicit operator, preprocess the entire query string and treat as implicit AND
    all_terms_processed = preprocess_query(query_string)
    # Wrap each stemmed term in its own list for consistency with 'terms' structure
    return {"terms": [[term] for term in all_terms_processed], "operator": None}


# Boolean Search Function
def boolean_search_with_operators(parsed_query, inverted_index):
    """Performs boolean search with AND, OR, NOT operators."""
    # Flatten the list of processed terms into a single list of stemmed terms
    query_terms = [t for sublist in parsed_query["terms"] for t in sublist if t]
    operator = parsed_query["operator"]

    if not query_terms:
        return []

    # Handle implicit AND (operator is None or an unrecognized operator for safety)
    if operator is None:
        # Perform implicit AND for all terms in query_terms
        if not query_terms:
            return []
        
        result_docs_set = set(inverted_index.get(query_terms[0], []))
        for term in query_terms[1:]:
            if not result_docs_set: # Optimization: if set is empty, no more intersections possible
                break
            result_docs_set = result_docs_set.intersection(set(inverted_index.get(term, [])))
        return sorted(list(result_docs_set))

    # Handle explicit binary operators (AND, OR, NOT)
    # The parse_boolean_query_string should ensure that for explicit operators,
    # `query_terms` will contain exactly two terms after flattening.
    if len(query_terms) == 2:
        term1_set = set(inverted_index.get(query_terms[0], []))
        term2_set = set(inverted_index.get(query_terms[1], []))

        if operator == "AND":
            result_docs = term1_set.intersection(term2_set)
        elif operator == "OR":
            result_docs = term1_set.union(term2_set)
        elif operator == "NOT":
            # "TERM1 NOT TERM2" means documents containing TERM1 but not TERM2
            result_docs = term1_set.difference(term2_set)
        else:
            # This case should ideally not be reached if operator is correctly identified
            result_docs = set()
        return sorted(list(result_docs))
    else:
        # This occurs if an explicit operator was given but the number of *processed* terms
        # after flattening was not 2. This implies a malformed query for binary operators.
        print(f"Warning: Malformed query for explicit operator '{operator}': expected 2 terms, found {len(query_terms)}. Returning no documents.")
        return []
