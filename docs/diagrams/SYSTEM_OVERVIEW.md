# Semantic Search Engine - System Flow Documentation

This document provides a comprehensive overview of the Semantic Search Engine's architecture and workflow through UML diagrams.

## Overview

The Semantic Search Engine is a sophisticated information retrieval system that uses natural language processing (NLP) and machine learning to understand the semantic meaning of search queries and match them with relevant documents. Unlike traditional keyword-based search, this system understands context and meaning.

## Architecture Components

### Core Components

1. **FastAPI Application** (`api.py`)
   - Web server and API endpoint handler
   - Orchestrates the entire search workflow
   - Manages data loading and caching

2. **Preprocessor** (`preprocessing.py`)
   - Text normalization and cleaning
   - Tokenization and lemmatization
   - Stopword removal

3. **Vectorizer** (`vectorization.py`)
   - Converts text to numerical embeddings
   - Uses Sentence-Transformers (MiniLM)
   - Generates 384-dimensional vectors

4. **Similarity Search** (`similarity_search.py`)
   - Computes cosine similarity
   - Compares query with document embeddings
   - Alternative FAISS implementation available

5. **Ranker** (`ranking.py`)
   - Sorts documents by relevance
   - Returns top-k results

6. **Utils** (`utils.py`)
   - Helper functions
   - Embedding generation and persistence

## System Flow

### High-Level Flow

```
User Query → Preprocessing → Vectorization → Similarity Search → Ranking → Results
```

### Detailed Flow

1. **Initialization**
   - Load product data from CSV
   - Check for existing document embeddings
   - Generate or load document embeddings

2. **Query Processing**
   - Receive user search query
   - Preprocess query (lowercase, clean, tokenize, lemmatize)
   - Convert to 384-dimensional embedding

3. **Search**
   - Compute cosine similarity between query and all documents
   - Generate similarity scores

4. **Ranking**
   - Sort documents by similarity score
   - Select top-5 most relevant results

5. **Response**
   - Gather product details for top results
   - Format and return JSON response

## UML Diagrams

### 1. Sequence Diagram
**File:** [SEQUENCE_DIAGRAM.md](SEQUENCE_DIAGRAM.md)

**Purpose:** Shows the chronological interaction between components during a search request.

**Key Insights:**
- Request/response flow between user and system
- Step-by-step preprocessing pipeline
- Component communication patterns
- Data transformations at each stage

**Best For:**
- Understanding how components interact
- Debugging flow issues
- Onboarding new developers

### 2. Class Diagram
**File:** [CLASS_DIAGRAM.md](CLASS_DIAGRAM.md)

**Purpose:** Illustrates the static structure of the system including classes and their relationships.

**Key Insights:**
- System components and their responsibilities
- Dependencies between modules
- Available methods and attributes
- External library integrations

**Best For:**
- Understanding system architecture
- Planning modifications or extensions
- Identifying component dependencies

### 3. Activity Diagram
**File:** [ACTIVITY_DIAGRAM.md](ACTIVITY_DIAGRAM.md)

**Purpose:** Depicts the workflow and decision logic from start to finish.

**Key Insights:**
- Complete process flow
- Decision points and conditions
- Parallel vs sequential operations
- System states and transitions

**Best For:**
- Understanding business logic
- Optimizing workflows
- Identifying bottlenecks

## Technology Stack

### Core Technologies
- **FastAPI**: Modern web framework for building APIs
- **Sentence Transformers**: Pre-trained models for text embeddings
- **NLTK**: Natural language processing toolkit
- **scikit-learn**: Machine learning utilities
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation

### Models
- **all-MiniLM-L6-v2**: Lightweight sentence transformer model
  - 384-dimensional embeddings
  - Fast inference
  - Good balance of speed and accuracy

### Similarity Metrics
- **Cosine Similarity**: Primary metric (sklearn)
- **FAISS**: Alternative for large-scale similarity search

## Key Features

### 1. Semantic Understanding
- Understands meaning beyond keywords
- Handles synonyms and related concepts
- Context-aware search

### 2. Efficient Processing
- Pre-computed document embeddings
- Fast similarity computation
- Cached models

### 3. Preprocessing Pipeline
- Lowercase normalization
- Special character removal
- Tokenization
- Stopword removal
- Lemmatization

### 4. Scalability
- FAISS support for large datasets
- Efficient vector operations
- Optimized embedding storage

## Performance Considerations

### Initialization Time
- Loading embeddings: Fast (NumPy load)
- Model loading: One-time cost
- CSV data loading: Minimal overhead

### Query Processing Time
1. **Preprocessing**: ~10-50ms
2. **Vectorization**: ~50-200ms (model dependent)
3. **Similarity Search**: ~1-10ms (depends on dataset size)
4. **Ranking**: <1ms
5. **Total**: ~100-300ms typical

### Memory Usage
- MiniLM model: ~80MB
- Document embeddings: ~4 bytes × 384 × num_documents
- Example: 10,000 products ≈ 15MB embeddings

## Use Cases

This semantic search engine is ideal for:
- E-commerce product search
- Document retrieval systems
- Content recommendation
- Question answering systems
- Knowledge base search

## Extending the System

### Adding New Features

1. **Alternative Vectorizers**
   - Modify `vectorization.py`
   - Add new model support
   - Update class diagram

2. **Different Similarity Metrics**
   - Extend `similarity_search.py`
   - Implement new metrics
   - Benchmark performance

3. **Advanced Ranking**
   - Enhance `ranking.py`
   - Add relevance feedback
   - Incorporate user preferences

4. **Caching Layer**
   - Add Redis/Memcached
   - Cache frequent queries
   - Update sequence diagram

## Troubleshooting

### Common Issues

1. **Slow First Query**
   - Cause: Model loading
   - Solution: Preload models at startup

2. **High Memory Usage**
   - Cause: Large embedding matrices
   - Solution: Use FAISS for compression

3. **Inaccurate Results**
   - Cause: Insufficient preprocessing
   - Solution: Tune preprocessing pipeline

## Best Practices

1. **Preprocessing**
   - Always preprocess consistently
   - Keep document and query preprocessing identical
   - Tune stopword list for domain

2. **Embeddings**
   - Generate embeddings offline when possible
   - Version control embedding parameters
   - Monitor embedding quality

3. **Search**
   - Adjust k based on use case
   - Consider hybrid search (semantic + keyword)
   - Implement relevance feedback

4. **Performance**
   - Profile regularly
   - Cache embeddings
   - Use batch processing for large datasets

## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [NLTK Documentation](https://www.nltk.org/)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [FAISS](https://github.com/facebookresearch/faiss)

## Contributing

When modifying the system:
1. Update relevant UML diagrams
2. Document new components
3. Update this overview
4. Add tests for new features
5. Benchmark performance impact

## Diagram Viewing

All diagrams use Mermaid syntax and can be viewed:
- **On GitHub**: Automatically rendered in markdown
- **VS Code**: Install Mermaid extension
- **Online**: Use [Mermaid Live Editor](https://mermaid.live/)

## Quick Reference

| Aspect | Details |
|--------|---------|
| **Framework** | FastAPI |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **Vector Dimension** | 384 |
| **Similarity Metric** | Cosine Similarity |
| **Top-K Results** | 5 (configurable) |
| **Preprocessing** | Lowercase, clean, tokenize, remove stopwords, lemmatize |
| **Storage Format** | NumPy (.npy) |

---

For detailed diagrams, visit:
- [Sequence Diagram](SEQUENCE_DIAGRAM.md)
- [Class Diagram](CLASS_DIAGRAM.md)
- [Activity Diagram](ACTIVITY_DIAGRAM.md)
