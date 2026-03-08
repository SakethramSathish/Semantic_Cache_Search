import numpy as np
from fastembed import TextEmbedding
from app.config import EMBEDDING_MODEL

# fastembed uses ONNX Runtime under the hood — no PyTorch, no DLL issues.
# "BAAI/bge-small-en-v1.5" is a drop-in replacement for all-MiniLM-L6-v2:
# same 384-dim output, same cosine-similarity semantics, faster on CPU.
_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = TextEmbedding(model_name=_MODEL_NAME)


def embed_texts(texts):
    """Embed a list of texts. Returns a numpy array of shape (N, 384)."""
    # fastembed returns a generator; convert to a stacked numpy array
    embeddings = list(_model.embed(texts))
    return np.array(embeddings, dtype="float32")


def embed_query(query: str):
    """Embed a single query string. Returns a 1-D numpy array of shape (384,)."""
    result = list(_model.embed([query]))
    return np.array(result[0], dtype="float32")
