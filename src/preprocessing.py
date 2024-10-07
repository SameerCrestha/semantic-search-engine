import nltk
# Add the local nltk_data folder to the NLTK path
nltk.data.path.append('nltk_data')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'\W+', ' ', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Example usage
if __name__ == "__main__":
    sample_text = "This is an Example Product Title!"
    processed_text = preprocess(sample_text)
    print("Original: {} \nFormatted: {}".format(sample_text,processed_text))
