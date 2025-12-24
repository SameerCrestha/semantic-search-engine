# Class Diagram - Semantic Search Engine

This class diagram shows the main components and their relationships in the semantic search engine.

```mermaid
classDiagram
    class FastAPIApp {
        +FastAPI app
        +DataFrame df
        +list~string~ product_titles
        +list~string~ preprocessed_texts
        +list~string~ product_urls
        +list~string~ product_prices
        +list~float~ ratings
        +list~int~ reviews
        +list~string~ manufacturers
        +ndarray document_embeddings
        +read_index() HTMLResponse
        +search(query string) dict
    }
    
    class Preprocessor {
        -WordNetLemmatizer lemmatizer
        -set~string~ stop_words
        +preprocess(text string) string
    }
    
    class Vectorizer {
        +tfidf_vectorization(corpus list) tuple
        +minilm_vectorization(corpus list) ndarray
    }
    
    class SentenceTransformer {
        -model SentenceTransformer
        +encode(texts list) ndarray
    }
    
    class SimilaritySearch {
        +cosine_similarity_search(query_embedding ndarray, document_embeddings ndarray) ndarray
        +faiss_search(query_embedding ndarray, document_embeddings ndarray) tuple
    }
    
    class Ranker {
        +rank_documents(similarity_scores ndarray, k int) ndarray
    }
    
    class Utils {
        +generate_and_save_embeddings(preprocessed_texts list) ndarray
    }
    
    class NLTKComponents {
        <<library>>
        +WordNetLemmatizer
        +word_tokenize()
        +stopwords
    }
    
    class SklearnComponents {
        <<library>>
        +TfidfVectorizer
        +cosine_similarity()
    }
    
    class FAISSIndex {
        <<library>>
        +IndexFlatL2
        +add()
        +search()
    }
    
    FastAPIApp --> Preprocessor : uses
    FastAPIApp --> Vectorizer : uses
    FastAPIApp --> SimilaritySearch : uses
    FastAPIApp --> Ranker : uses
    FastAPIApp --> Utils : uses
    
    Preprocessor --> NLTKComponents : depends on
    Vectorizer --> SentenceTransformer : uses
    Vectorizer --> SklearnComponents : uses
    SimilaritySearch --> SklearnComponents : uses
    SimilaritySearch --> FAISSIndex : uses
    Utils --> Vectorizer : uses
    
    note for FastAPIApp "Main application entry point\nHandles HTTP requests and\norchestrates search workflow"
    note for Preprocessor "Text preprocessing pipeline\nNormalizes and cleans text data"
    note for Vectorizer "Converts text to numerical vectors\nSupports TF-IDF and MiniLM"
    note for SimilaritySearch "Computes similarity between\nquery and document vectors"
```

## Component Descriptions

### FastAPIApp
- **Purpose**: Main application server that orchestrates the search workflow
- **Responsibilities**: 
  - Serves HTTP endpoints
  - Loads and manages document embeddings
  - Coordinates between preprocessing, vectorization, search, and ranking
  - Returns formatted search results

### Preprocessor
- **Purpose**: Prepares text for vectorization
- **Responsibilities**:
  - Lowercase conversion
  - Special character removal
  - Tokenization
  - Stopword removal
  - Lemmatization

### Vectorizer
- **Purpose**: Converts text to numerical representations
- **Responsibilities**:
  - TF-IDF vectorization (alternative method)
  - MiniLM sentence embeddings (primary method)
  - Returns 384-dimensional vectors

### SimilaritySearch
- **Purpose**: Compares query vectors with document vectors
- **Responsibilities**:
  - Cosine similarity computation
  - FAISS-based nearest neighbor search (alternative method)

### Ranker
- **Purpose**: Ranks documents based on similarity scores
- **Responsibilities**:
  - Sorts similarity scores
  - Returns top-k most relevant documents

### Utils
- **Purpose**: Helper functions for the application
- **Responsibilities**:
  - Generate and save document embeddings
  - Utility operations
