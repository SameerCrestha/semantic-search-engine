# UML Diagrams - Semantic Search Engine

This directory contains comprehensive UML diagrams that describe the flow and architecture of the Semantic Search Engine.

## 📖 Documentation

- **[System Overview](SYSTEM_OVERVIEW.md)** - Complete system documentation including architecture, flow, performance, and best practices

## Available Diagrams

### 1. [Sequence Diagram](SEQUENCE_DIAGRAM.md)
Shows the step-by-step interaction between components during a search request.

**Key aspects:**
- Request/response flow
- Component interactions
- Data transformations
- Processing sequence

**Use this diagram to understand:**
- How a search query flows through the system
- Which components interact at each step
- The sequence of operations

### 2. [Class Diagram](CLASS_DIAGRAM.md)
Illustrates the structure of the system including classes, their attributes, methods, and relationships.

**Key aspects:**
- System components
- Class relationships
- Dependencies
- Methods and attributes

**Use this diagram to understand:**
- The overall architecture
- Component dependencies
- Available methods and operations
- System structure

### 3. [Activity Diagram](ACTIVITY_DIAGRAM.md)
Depicts the workflow of the search process from start to finish.

**Key aspects:**
- Process flow
- Decision points
- Parallel activities
- Workflow phases

**Use this diagram to understand:**
- The complete search workflow
- Initialization vs. runtime processes
- Decision logic
- Process steps

## Diagram Format

All diagrams are created using [Mermaid](https://mermaid.js.org/), which is:
- Natively supported by GitHub
- Easy to version control
- Text-based and maintainable
- Automatically rendered in GitHub markdown

## How to View

Simply click on any diagram file above, and GitHub will automatically render the Mermaid diagrams.

Alternatively, you can:
1. Use any Mermaid-compatible viewer
2. Use VS Code with Mermaid extension
3. Use online Mermaid editors like [Mermaid Live Editor](https://mermaid.live/)

## System Overview

The Semantic Search Engine uses the following workflow:

1. **Initialization**: Load product data and document embeddings
2. **Query Processing**: Receive and preprocess user query
3. **Vectorization**: Convert query to numerical embedding
4. **Similarity Search**: Compare query embedding with document embeddings
5. **Ranking**: Rank documents by similarity scores
6. **Response**: Return top-k most relevant results

## Technologies Represented

- **FastAPI**: Web framework for API endpoints
- **NLTK**: Natural language processing
- **Sentence Transformers**: Text embedding generation
- **scikit-learn**: Cosine similarity computation
- **FAISS**: Fast similarity search (alternative method)
- **NumPy**: Numerical operations
- **Pandas**: Data management

## Quick Reference

| Diagram | Best For | Key Information |
|---------|----------|-----------------|
| Sequence | Understanding flow | Component interactions, request/response cycle |
| Class | Understanding structure | Components, relationships, methods |
| Activity | Understanding process | Workflow, decision points, phases |

## Contributing

If you update the system architecture or workflow, please update the relevant diagrams to keep documentation in sync with code.
