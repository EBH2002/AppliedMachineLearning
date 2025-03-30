from sklearn.base import BaseEstimator as ModelType
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))

model = joblib.load(r"C:\Users\HP\model.pkl")

vectorizer = joblib.load(r"C:\Users\HP\tfidf_vectorizer.pkl")

def clean_text(input_text: str) -> str:
    """
    Preprocesses the input text by tokenizing, removing stopwords, lemmatizing,
    converting to lowercase, and filtering empty strings.

    Args:
        input_text (str): The raw text to be processed.

    Returns:
        str: A cleaned and space-separated string ready for vectorization.
    """
    # Tokenizing the text
    words = word_tokenize(input_text)

    # Removing stopwords
    words = [word for word in words if word not in STOPWORDS]

    # Applying lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Converting text to lowercase
    words = [word.lower() for word in words if word]

    return " ".join(words)

def score(text: str, classifier, threshold: float = 0.5) -> tuple[bool, float]:
    """
    Evaluates a given text using a trained model and returns a classification result
    along with the confidence score.

    Args:
        text (str): The input text for classification.
        classifier (ModelType): The trained model for scoring.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer used to transform text.
        threshold (float): Decision threshold for classification (default: 0.5).

    Returns:
        tuple[bool, float]: A tuple containing the classification result and the confidence score.
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    if not (0 <= threshold <= 1):
        raise ValueError("Threshold should be within the range [0, 1].")

    # Preprocess the input text
    processed_text = clean_text(text)

    # Transform the text using the vectorizer
    text_vectorized = vectorizer.transform([processed_text]).toarray()

    # Compute confidence score (propensity)
    confidence_score = classifier.predict_proba(text_vectorized)[0][1]

    # Determine final prediction based on the threshold
    is_positive = confidence_score > threshold

    return bool(is_positive), float(confidence_score)
