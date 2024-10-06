from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# TF-IDF vectorization function
def tfidf_vectorization(corpus):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

# MiniLM embedding function
def minilm_vectorization(corpus):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    document_embeddings = model.encode(corpus)
    return document_embeddings

# Example usage
if __name__ == "__main__":
    corpus = ["This is a sample sentence.", "Another sample sentence."]
    tfidf_matrix, vectorizer = tfidf_vectorization(corpus)
    print(tfidf_matrix.toarray())

    minilm_embeddings = minilm_vectorization(corpus)
    print(minilm_embeddings)
