import faiss
import numpy as np
import pandas as pd

def build_index(embeddings_file, output_dir):
    data = pd.read_pickle(embeddings_file)
    embeddings = np.vstack(data['embeddings'].values)

    # Create and save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f"{output_dir}/faiss_index")

if __name__ == "__main__":
    build_index("../embeddings/embeddings.pkl", "../embeddings/faiss_index")
