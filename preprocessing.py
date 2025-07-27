import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
# Initialize stemmer
ps = PorterStemmer()

def transform_text(text):
    # Lowercase
    text = text.lower()

    # Tokenize
    words = word_tokenize(text)

    # Keep only alphanumeric
    words = [word for word in words if word.isalnum()]

    # Remove stopwords and punctuation
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]

    # Stemming
    words = [ps.stem(word) for word in words]

    return " ".join(words)
