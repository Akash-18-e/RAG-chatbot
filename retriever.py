import faiss
import pickle
import numpy as np

def load_index():
    index = faiss.read_index("faiss_index.bin")

    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def search(query_embedding, k=3):
    index, chunks = load_index()

    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = [chunks[i] for i in indices[0]]

    return results