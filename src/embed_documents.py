from sentence_transformers import SentenceTransformer
import pandas as pd

def generate_embeddings(model_name, input_file, output_dir):
    model = SentenceTransformer(model_name)
    data = pd.read_csv(input_file)

    # Generate embeddings
    data['embeddings'] = data['text'].apply(lambda x: model.encode(x))
    data.to_pickle(f"{output_dir}/embeddings.pkl")

if __name__ == "__main__":
    generate_embeddings("all-MiniLM-L6-v2", "../data/legal_documents.csv", "../embeddings")
