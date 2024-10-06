import numpy as np

# Function to rank documents based on similarity scores
def rank_documents(similarity_scores, k=5):
    ranked_indices = np.argsort(similarity_scores[0])[-k:][::-1]  # Top k results
    return ranked_indices

# Example usage
if __name__ == "__main__":
    similarity_scores = np.random.rand(1, 10)  # Example similarity scores
    top_results = rank_documents(similarity_scores, k=5)
    print("Top-ranked document indices:", top_results)
