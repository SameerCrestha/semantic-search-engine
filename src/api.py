import os
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .preprocessing import preprocess
from .vectorization import minilm_vectorization
from .similarity_search import cosine_similarity_search
from .ranking import rank_documents

app = FastAPI()

# Define the absolute path to the static folder
static_folder_path = os.path.abspath("../static")

# Mount the static files directory
app.mount("/static", StaticFiles(directory=static_folder_path), name="static")

# Load CSV data 
df = pd.read_csv("../data/processed_data.csv")  

# Extract the necessary columns
product_titles = df['Product Name'].tolist()
preprocessed_texts = df['processed_text'].tolist()
product_urls = df['Product URL'].tolist()
product_prices = df['Product Price'].tolist()
ratings = df['Rating'].tolist()
reviews = df['Number of reviews'].tolist()
manufacturers = df['Manufacturer'].tolist()

# Vectorize the processed_text column for similarity search
document_embeddings = minilm_vectorization(preprocessed_texts)

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(static_folder_path, "index.html")
    try:
        with open(index_path, "r") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="Index file not found.", status_code=404)

@app.get("/search/")
def search(query: str):
    # Preprocess the query
    query_processed = preprocess(query)
    
    # Vectorize the query
    query_embedding = minilm_vectorization([query_processed])
    
    # Perform similarity search
    similarity_scores = cosine_similarity_search(query_embedding, document_embeddings)
    
    # Rank the documents based on similarity
    top_results_indices = rank_documents(similarity_scores)
    
    # Prepare the result with product details
    results = [
        {
            "product_title": product_titles[idx],
            "product_url": product_urls[idx],
            "product_price": product_prices[idx],
            "rating": ratings[idx],
            "reviews": reviews[idx],
            "manufacturer": manufacturers[idx],
            "similarity_score": round(float(similarity_scores[0][idx]), 2)  # Convert to float and round
        }
        for idx in top_results_indices
    ]
    # Return the results
    return {
        "results": results  # List of dictionaries with all product details
    }

# Run with: fastapi run src/api.py
