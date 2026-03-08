from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import joblib
import json
import numpy as np

# Import our custom semantic cache
from semantic_cache import SemanticCache

app = FastAPI(title="Lightweight Semantic Search Cache API")

# Global instances
cache = None
faiss_index = None
docs = []
metadata = []

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None = None
    similarity_score: float | None = None
    result: dict
    dominant_cluster: int

@app.on_event("startup")
async def startup_event():
    global cache, faiss_index, docs, metadata
    print("Loading data and initializing cache...")
    try:
        # Load the base dataset for raw retrieval
        with open('data/documents.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            docs = data['docs']
            metadata = data['metadata']
            
        # Initialize the custom Semantic Cache
        cache = SemanticCache(threshold=0.85)
        
        # Load FAISS index for actual search when cache misses
        faiss_index = faiss.read_index('data/faiss_index.bin')
        print("Initialization complete.")
    except Exception as e:
        print(f"Failed to load data on startup: {e}")
        print("Please ensure data_pipeline.py has been run successfully.")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not cache or not faiss_index:
        raise HTTPException(status_code=500, detail="System not initialized. Run data_pipeline.py first.")
        
    # Check Semantic Cache first
    cache_result = cache.search(req.query)
    
    if cache_result["cache_hit"]:
        return {
            "query": req.query,
            "cache_hit": True,
            "matched_query": cache_result["matched_query"],
            "similarity_score": cache_result["similarity_score"],
            "result": cache_result["result"],
            "dominant_cluster": cache_result["dominant_cluster"]
        }
        
    # ------------- CACHE MISS PATH -------------
    # We use the embedding and probabilities already computed by the cache to save time
    emb = cache_result["query_embedding"]
    probs = cache_result["cluster_probs"]
    dominant_cluster = cache_result["dominant_cluster"]
    
    # Perform raw search against FAISS index
    k = 5 # Return top 5 matches
    scores, indices = faiss_index.search(emb, k)
    
    # Construct response
    top_matches = []
    for score, idx in zip(scores[0], indices[0]):
        top_matches.append({
            "doc": docs[idx][:500] + "..." if len(docs[idx]) > 500 else docs[idx], # truncate for readability
            "category": metadata[idx]["category"],
            "similarity": float(score)
        })
        
    response_payload = {"top_matches": top_matches}
    
    # Store the result in our custom Semantic Cache
    cache.add(
        query=req.query,
        embedding=emb,
        probs=probs,
        dominant_cluster=dominant_cluster,
        response=response_payload
    )
    
    return {
        "query": req.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": response_payload,
        "dominant_cluster": dominant_cluster
    }

@app.get("/cache/stats")
async def get_cache_stats():
    if not cache:
        raise HTTPException(status_code=500, detail="Cache not initialized.")
    return cache.get_stats()

@app.delete("/cache")
async def flush_cache():
    if not cache:
        raise HTTPException(status_code=500, detail="Cache not initialized.")
    cache.flush()
    return {"status": "success", "message": "Cache flushed entirely."}
