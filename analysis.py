import os
import json
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
DB_PATH = 'data/faiss_index.bin'
DOCS_PATH = 'data/documents.json'
GMM_MODEL_PATH = 'data/gmm_model.joblib'

def analyze_clusters():
    """
    Demonstrates fuzzy clustering assignments for the 20 newsgroups dataset.
    This script finds:
    1. A 'Core' document for a cluster (very high probability in one cluster).
    2. A 'Boundary' document (probability spread across multiple clusters, uncertainty).
    """
    print("Loading models and data for analysis...")
    if not os.path.exists(GMM_MODEL_PATH) or not os.path.exists(DOCS_PATH):
        print("Required data files not found. Please run data_pipeline.py first.")
        return

    gmm = joblib.load(GMM_MODEL_PATH)
    
    with open(DOCS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        docs = data['docs']
        metadata = data['metadata']
        
    print(f"Loaded {len(docs)} documents.")
    
    # We will just embed a small sample to find examples instead of the whole dataset 
    # to save time in the analysis script, or we could load all embeddings (but we didn't save them separately).
    # Since we need to find specific examples, let's embed a random subset of 1000 docs.
    np.random.seed(42)
    sample_indices = np.random.choice(len(docs), size=min(1000, len(docs)), replace=False)
    sample_docs = [docs[i] for i in sample_indices]
    sample_meta = [metadata[i] for i in sample_indices]
    
    print("Embedding sample documents...")
    model = SentenceTransformer(MODEL_NAME)
    sample_embeddings = model.encode(sample_docs, show_progress_bar=False)
    sample_embeddings = np.array(sample_embeddings).astype('float32')
    faiss.normalize_L2(sample_embeddings)
    
    print("Predicting fuzzy cluster distributions...")
    # predict_proba returns a matrix of shape (n_samples, n_components)
    # where each row sums to 1 (the probability distribution over clusters)
    probs = gmm.predict_proba(sample_embeddings)
    
    # Find a CORE document (max probability > 0.95)
    max_probs = np.max(probs, axis=1)
    core_idx = np.argmax(max_probs)
    
    # Find a BOUNDARY document (genuine uncertainty, max probability is low, e.g., < 0.3)
    # We want a document where the probability is heavily split.
    # Entropy could also be used, but lowest max probability is simple and effective.
    boundary_idx = np.argmin(max_probs)
    
    print("\n" + "="*50)
    print("CORE CLUSTER EXAMPLE (High Certainty)")
    print("="*50)
    print(f"Top cluster probability: {max_probs[core_idx]:.4f}")
    print(f"Assigned Cluster: {np.argmax(probs[core_idx])}")
    print(f"Original Category: {sample_meta[core_idx]['category']}")
    print("-" * 20)
    print(f"Snippet:\n{sample_docs[core_idx][:500]}...")
    
    print("\n" + "="*50)
    print("BOUNDARY EXAMPLE (Genuine Uncertainty)")
    print("="*50)
    print(f"Top 3 cluster probabilities:")
    top3_clusters = np.argsort(probs[boundary_idx])[::-1][:3]
    for c in top3_clusters:
        print(f"  Cluster {c}: {probs[boundary_idx][c]:.4f}")
    print(f"Original Category: {sample_meta[boundary_idx]['category']}")
    print("-" * 20)
    print(f"Snippet:\n{sample_docs[boundary_idx][:500]}...")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    analyze_clusters()
