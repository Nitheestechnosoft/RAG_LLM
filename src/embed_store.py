import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def create_faiss_index(chunks):
    #loading the embedding model which converts text to vector numbers
    model = SentenceTransformer("all-MiniLM-L6-v2")

    #encoding chunks
    embeddings = model.encode(chunks)

    #faiss model needs vectors to stored in float32 type, we are converting the type in numpy
    embeddings =  np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]

    #learning method for finding the distance FlatL2 is simple vector organization
    index = faiss.IndexFlatL2(dimension)

    #storing all the vectors in FAISS
    index.add(embeddings)

    return index, model
