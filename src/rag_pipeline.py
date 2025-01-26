import os
import json
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up directories and file paths
preprocessed_data_dir = "../dataset/processed-IN-Ext/"
train_file_A1 = os.path.join(preprocessed_data_dir, "full_summaries_A1.jsonl")

# Load the fine-tuned model and tokenizer
model_path = "../fine_tuned_lora_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model is running on: {model.device}")

# Load and preprocess the dataset
def load_dataset_for_retrieval(jsonl_file):
    """
    Load preprocessed data and extract judgments for retrieval.
    """
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Extract judgments and deduplicate
    judgments = list(set(item["judgement"].strip() for item in data))
    return judgments

# Load judgments from the file and deduplicate
all_judgments = load_dataset_for_retrieval(train_file_A1)

# Convert judgments to TF-IDF vectors for retrieval
vectorizer = TfidfVectorizer()
judgment_vectors = vectorizer.fit_transform(all_judgments).toarray().astype(np.float32)

# Build FAISS index for efficient similarity search
index = faiss.IndexFlatL2(judgment_vectors.shape[1])  # L2 distance
index.add(judgment_vectors)

def retrieve_judgments(query, top_k=3):
    """
    Retrieve top-k judgments relevant to the query using FAISS.
    """
    query_vector = vectorizer.transform([query]).toarray().astype(np.float32)
    distances, indices = index.search(query_vector, top_k)
    return [all_judgments[i] for i in indices[0]]

# Implement the RAG pipeline
def pipeline(query, top_k=3, max_length=4096):
    """
    RAG pipeline to retrieve judgments and generate summaries using the fine-tuned model.
    """
    # Retrieve relevant judgments
    retrieved_judgments = retrieve_judgments(query, top_k=top_k)
    
    summaries = []
    for judgment in retrieved_judgments:
        # Format the input using a prompt template
        input_text = f"### Instruction: Summarize the following legal text.\n\n### Input:\n{judgment.strip()[:10000]}\n\n### Response:\n".strip()
        
        # Tokenize and move inputs to GPU
        inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Generate summary
        outputs = model.generate(**inputs, max_new_tokens=1000)
        
        # Decode the generated summary
        summary = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        summaries.append(summary)
    
    return summaries

if __name__ == "__main__":
    query = "lakshminarayana iyer"
    summaries = pipeline(query, top_k=1)

    print("Retrieved Summaries:")
    for i, summary in enumerate(summaries):
        print(f"Summary {i+1}:\n{summary}\n")