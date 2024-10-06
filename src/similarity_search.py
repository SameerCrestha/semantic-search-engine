from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

# Cosine similarity function
def cosine_similarity_search(query_embedding, document_embeddings):
    similarity_scores = cosine_similarity(query_embedding, document_embeddings)
    return similarity_scores

# FAISS-based nearest neighbor search
def faiss_search(query_embedding, document_embeddings):
    index = faiss.IndexFlatL2(document_embeddings.shape[1])  # Dimensionality must match
    index.add(document_embeddings)
    D, I = index.search(query_embedding, k=5)  # Search for top-5 neighbors
    return D, I

# Example usage
if __name__ == "__main__":
    query_embedding = np.random.rand(1, 384)  # Example query embedding
    document_embeddings = np.random.rand(100, 384)  # Example document embeddings
    
    # Cosine similarity search
    cosine_scores = cosine_similarity_search(query_embedding, document_embeddings)
    print(cosine_scores)

    # FAISS search
    distances, indices = faiss_search(query_embedding, document_embeddings)
    print(indices)
