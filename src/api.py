from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import rag_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query(request: QueryRequest):
    response = rag_pipeline(
        request.query, 
        "../models/llama2-finetuned/", 
        "../embeddings/faiss_index", 
        "../data/legal_documents.csv"
    )
    return {"response": response}
