import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

# Implement the clean(text) function
def clean(text):
    # Case folding (lowercase conversion)
    text = text.lower()
    # Normalize text by removing numbers and punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Implement the tokenize(text) function
def tokenize(text):
    # Use word_tokenize to split the cleaned text into a list of tokens
    return word_tokenize(text)

# Implement the remove_stopwords(tokens) function
def remove_stopwords(tokens):
    # Filter out stopwords from the list of tokens
    return [word for word in tokens if word not in indonesian_stopwords]

# Implement the stem(tokens) function
def stem(tokens):
    # Apply stemming to each token in the list using PorterStemmer
    return [ps.stem(word) for word in tokens]
