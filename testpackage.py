print("Starting package test...")

# Test FAISS
import faiss
print("FAISS OK")

# Test PDF reader
import pypdf
print("PyPDF OK")

# Test embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer OK")

# Quick embedding test
vector = model.encode("Testing embeddings")
print("Embedding vector length:", len(vector))

print("All packages are working correctly")
