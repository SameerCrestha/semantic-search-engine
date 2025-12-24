# Sequence Diagram - Semantic Search Engine

This sequence diagram shows the flow of a search request through the semantic search engine system.

```mermaid
sequenceDiagram
    participant User
    participant FastAPI as FastAPI Server
    participant Preprocessor
    participant Vectorizer
    participant SimilaritySearch as Similarity Search
    participant Ranker
    participant Database as Document Store
    
    User->>FastAPI: GET /search?query=laptop
    FastAPI->>Database: Load document embeddings
    Database-->>FastAPI: Return document_embeddings.npy
    
    FastAPI->>Preprocessor: preprocess(query)
    Preprocessor->>Preprocessor: Convert to lowercase
    Preprocessor->>Preprocessor: Remove special characters
    Preprocessor->>Preprocessor: Tokenize words
    Preprocessor->>Preprocessor: Remove stopwords
    Preprocessor->>Preprocessor: Lemmatize tokens
    Preprocessor-->>FastAPI: Return processed query
    
    FastAPI->>Vectorizer: minilm_vectorization(processed_query)
    Vectorizer->>Vectorizer: Load SentenceTransformer model
    Vectorizer->>Vectorizer: Generate 384-dim embedding
    Vectorizer-->>FastAPI: Return query_embedding
    
    FastAPI->>SimilaritySearch: cosine_similarity_search(query_embedding, document_embeddings)
    SimilaritySearch->>SimilaritySearch: Calculate cosine similarity
    SimilaritySearch->>SimilaritySearch: Compare query vs all documents
    SimilaritySearch-->>FastAPI: Return similarity_scores
    
    FastAPI->>Ranker: rank_documents(similarity_scores, k=5)
    Ranker->>Ranker: Sort by similarity scores
    Ranker->>Ranker: Get top-k indices
    Ranker-->>FastAPI: Return top_results_indices
    
    FastAPI->>FastAPI: Build results with product details
    FastAPI-->>User: Return JSON results with products
```

## Flow Description

1. **User Request**: User sends a search query through the GET endpoint
2. **Load Embeddings**: FastAPI loads pre-computed document embeddings from storage
3. **Preprocessing**: Query text is cleaned and normalized:
   - Convert to lowercase
   - Remove special characters
   - Tokenize into words
   - Remove stopwords (common words like "the", "is", etc.)
   - Lemmatize (reduce words to base form)
4. **Vectorization**: Processed query is converted to 384-dimensional embedding using MiniLM model
5. **Similarity Search**: Query embedding is compared against all document embeddings using cosine similarity
6. **Ranking**: Documents are ranked by similarity scores and top-k results are selected
7. **Response**: Results are formatted with product details and returned to user
