import faiss
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer

def rag_pipeline(query, model_name, index_file, docs_file):
    # Load model, tokenizer, and FAISS index
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    index = faiss.read_index(index_file)
    data = pd.read_pickle(docs_file)

    # Query embedding
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k=5)

    # Retrieve context
    context = " ".join([data.iloc[idx]['text'] for idx in indices[0]])
    input_text = f"Context: {context}\n\nQuestion: {query}"

    # Generate response
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs.input_ids)
    return tokenizer.decode(outputs[0])

if __name__ == "__main__":
    query = "What is the penalty clause in an NDA?"
    response = rag_pipeline(query, "../models/llama2-finetuned/", "../embeddings/faiss_index", "../data/legal_documents.csv")
    print(response)
