from .vectorization import minilm_vectorization
import numpy as np
def generate_and_save_embeddings(preprocessed_texts):
    embeddings = minilm_vectorization(preprocessed_texts)
    np.save('data/document_embeddings.npy', embeddings)
    print("Document embeddings generated and saved.")
    return embeddings