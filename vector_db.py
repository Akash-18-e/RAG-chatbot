import faiss
import numpy as np
import pickle

def store_embeddings(chunks, embeddings):
    # Convert to numpy array
    vectors = np.array(embeddings).astype("float32")

    # Get dimension
    dimension = vectors.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)

    # Add vectors
    index.add(vectors)

    # Save index
    faiss.write_index(index, "faiss_index.bin")

    # Save chunks separately (important!)
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Embeddings stored in FAISS")