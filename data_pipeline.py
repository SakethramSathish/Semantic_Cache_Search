import os
import json
import numpy as np
import faiss
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture
import joblib

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
DB_PATH = 'data/faiss_index.bin'
DOCS_PATH = 'data/documents.json'
GMM_MODEL_PATH = 'data/gmm_model.joblib'
N_CLUSTERS = 20 # Aligning with 20 newsgroups

def ensure_dirs():
    if not os.path.exists('data'):
        os.makedirs('data')

def load_and_clean_data():
    """
    Fetches the 20 newsgroups dataset and performs deliberate data cleaning.
    
    Data Cleaning Decisions:
    1. remove=('headers', 'footers', 'quotes'): We explicitly remove these sections.
       - Headers contain metadata (From, Subject, Organization, Lines) which can cause the model
         to cluster based on metadata rather than the semantic content of the discussion.
       - Footers are often signatures.
       - Quotes (lines starting with >) are text from previous emails in a thread. Including them
         causes the model to embed redundant information and can dilute the unique semantic contribution
         of the current post.
    2. Length validation: We discard posts that become too short (< 20 characters) after removing
       headers/footers/quotes, as they lack sufficient semantic context to be useful for retrieval.
    """
    print("Fetching 20 newsgroups dataset...")
    # Fetching only the 'train' subset for the corpus to keep it manageable,
    # and removing headers, footers, and quotes for true semantic representation.
    dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    
    raw_docs = dataset.data
    targets = dataset.target
    target_names = dataset.target_names
    
    cleaned_docs = []
    metadata = []
    
    print("Cleaning data...")
    for doc, target_idx in zip(raw_docs, targets):
        # Strip leading/trailing whitespace
        doc_clean = doc.strip()
        
        # Keep documents with at least 20 characters.
        # Justification: Extremely short documents (e.g. "Yes.", "I agree.") do not contain
        # enough semantic information to be meaningfully embedded or retrieved in a search context.
        if len(doc_clean) >= 20:
            cleaned_docs.append(doc_clean)
            metadata.append({
                "category": target_names[target_idx]
            })
            
    print(f"Retained {len(cleaned_docs)} out of {len(raw_docs)} documents after cleaning.")
    return cleaned_docs, metadata

def create_embeddings(docs):
    """
    Generates dense vector embeddings for the documents.
    
    Embedding Model Choice: 'sentence-transformers/all-MiniLM-L6-v2'
    Justification:
    - This model is specifically tuned for general-purpose semantic similarity.
    - It's lightweight and fast, making it ideal for a responsive cache system.
    - It produces 384-dimensional vectors, offering a great balance between
      expressive power and computational/storage efficiency (compared to e.g., 768d or 1536d models),
      which is crucial for fast FAISS lookups as well as building the GMM quickly.
    """
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Generating embeddings for {len(docs)} documents...")
    # Generate embeddings. The output is a numpy array of shape (N, 384)
    embeddings = model.encode(docs, show_progress_bar=True)
    return embeddings

def build_vector_store(embeddings):
    """
    Builds and persists a FAISS index.
    
    Vector Store Choice: FAISS (Facebook AI Similarity Search)
    Justification:
    - FAISS is exceptionally fast for vector similarity search, which is exactly what our
      downstream retrieval needs.
    - We use IndexFlatIP (Inner Product). Since our SentenceTransformer embeddings are
      typically normalized (or effectively measuring cosine similarity via dot product),
      Inner Product is highly efficient for cosine similarity search.
    - It is very easy to serialize to disk, fulfilling the persistence requirement.
    """
    print("Building FAISS index...")
    # Dimension of the embeddings
    dim = embeddings.shape[1]
    
    # We use exact search via inner product.
    # For normalized vectors, inner product is equal to cosine similarity.
    index = faiss.IndexFlatIP(dim)
    
    # SentenceTransformers embeddings are typically NOT L2 normalized by default, 
    # but cosine similarity is standard. We can normalize vectors before adding
    # if we strictly want IP to represent cosine similarity.
    # L2 normalization for proper cosine similarity via Inner Product
    faiss.normalize_L2(embeddings)
    
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

def build_fuzzy_clusters(embeddings):
    """
    Builds a Gaussian Mixture Model (GMM) for fuzzy clustering.
    
    Clustering Choice: Gaussian Mixture Model (GMM)
    Justification:
    - Unlike K-Means, which assigns a hard label to each point, GMM models the data as a
      mixture of multiple Gaussian distributions.
    - This allows us to compute `.predict_proba()`, which gives us a distribution of
      probabilities across all clusters for any given document/query. This fulfills
      the requirement that output is a distribution, not a single label.
    
    Number of Clusters (N_CLUSTERS=20):
    - The underlying 20 newsgroups dataset inherently has 20 categories.
    - Setting N_CLUSTERS to 20 allows the GMM a reasonable structural capacity to capture
      these latent topics in the vector space.
    - In a real-world unlabelled scenario, one might use BIC (Bayesian Information Criterion)
      or silhouette scores to sweep over N_CLUSTERS, but here we have a strong semantic prior.
    """
    print(f"Training Gaussian Mixture Model for fuzzy clustering with {N_CLUSTERS} components...")
    # Setting covariance_type='spherical' to speed up training in high dimensional space (384d),
    # and preventing singular covariance matrices if dimensions are correlated.
    # Using a subset or full embedding for training
    gmm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='spherical', random_state=42, verbose=1)
    
    # We use the normalized embeddings so the distribution is modeled on the unit hypersphere.
    gmm.fit(embeddings)
    print("GMM training completed.")
    return gmm

def main():
    ensure_dirs()
    
    # Step 1: Load and clean
    docs, metadata = load_and_clean_data()
    
    # Step 2: Embed
    embeddings = create_embeddings(docs)
    
    # Convert embeddings to float32 as required by FAISS
    embeddings = np.array(embeddings).astype('float32')
    
    # We make a copy of normalized embeddings for FAISS and GMM
    normalized_embeddings = embeddings.copy()
    faiss.normalize_L2(normalized_embeddings)
    
    # Step 3: Vector Store
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    
    # Step 4: Fuzzy Clustering
    gmm = build_fuzzy_clusters(normalized_embeddings)
    
    # Step 5: Persist everything
    print("Persisting data offline...")
    faiss.write_index(index, DB_PATH)
    joblib.dump(gmm, GMM_MODEL_PATH)
    
    # Save documents and metadata for retrieval
    with open(DOCS_PATH, 'w', encoding='utf-8') as f:
        json.dump({"docs": docs, "metadata": metadata}, f)
        
    print("Data pipeline executed successfully!")

if __name__ == "__main__":
    main()
