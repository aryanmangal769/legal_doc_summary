# Legal Document Summarization with Fine-Tuned LLM and RAG Pipeline

This project focuses on two main objectives:
1. **Fine-tuning a Large Language Model (LLM)** for summarizing legal documents.
2. **Implementing a Retrieval-Augmented Generation (RAG) pipeline** to automatically retrieve and summarize legal documents based on user-provided details.

The project leverages the **Llama-2-7b** model, fine-tuned using **LoRA (Low-Rank Adaptation)**, and integrates a RAG pipeline for efficient document retrieval and summarization.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Dataset Preparation](#dataset-preparation)
4. [Fine-Tuning the LLM](#fine-tuning-the-llm)
5. [RAG Pipeline](#rag-pipeline)
6. [Running the Project](#running-the-project)
7. [Future Work](#future-work)
8. [References](#references)

---

## Project Overview

### 1. Fine-Tuning LLM for Legal Summarization
- The project fine-tunes the **Llama-2-7b** model using **LoRA** to adapt it for summarizing legal documents.
- The model is trained on a dataset of legal judgments and their corresponding summaries.
- The training process uses the **SFTTrainer** from the `trl` library, which simplifies fine-tuning with LoRA.

### 2. RAG Pipeline for Document Retrieval
- The RAG pipeline retrieves relevant legal documents based on user queries (e.g., case names or details).
- It uses **FAISS** for efficient similarity search and **TF-IDF** for document vectorization.
- The retrieved documents are then summarized using the fine-tuned LLM.

---

## Setup Instructions

### 1. Create a Conda Environment
```bash
conda create --name legal_assistant python=3.10
conda activate legal_assistant
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Congifurtions for using huggingfac llama model  Llama-2-7b Model
1. Visit the [Llama 2 hugging-face page](https://huggingface.co/meta-llama/Llama-2-7b-hf/) and request access to the model.
2. Once approved, log in to Hugging Face:
   ```bash
   huggingface-cli login
   ```


### 4. Install Additional Dependencies
```bash
pip install sentencepiece datasets trl bitsandbytes faiss-cpu
```

---

## Dataset Preparation

### 1. Download the Dataset
- Download the dataset from [Zenodo](https://zenodo.org/records/7152317#.ZCSfaoTMI2y).
- Extract the dataset into the `legal-llm-project/datasets` directory.

### 2. Preprocess the Dataset
Run the preprocessing script to prepare the dataset for training:
```bash
python src/data_preprocessing.py
```

---

## Fine-Tuning the LLM

### 1. Configure LoRA
- LoRA is used to fine-tune the Llama-2-7b model with a low-rank adaptation approach.
- The configuration includes parameters like `lora_alpha`, `lora_dropout`, and `r` (rank).

### 2. Training with SFTTrainer
- The `SFTTrainer` from the `trl` library is used for fine-tuning.
- The dataset is formatted with clear distinctions between instructions, input, and response:
  ```text
  ### Instruction: Summarize the following legal text.

  ### Input:
  {legal_text}

  ### Response:
  {summary}
  ```

### 3. Save the Fine-Tuned Model
After training, the fine-tuned model is saved for inference:
```bash
model.save_pretrained("../fine_tuned_lora_model")
tokenizer.save_pretrained("../fine_tuned_lora_model")
```

---

## RAG Pipeline

### 1. Document Retrieval
- The pipeline uses **FAISS** for efficient similarity search.
- Documents are vectorized using **TF-IDF** for retrieval.

### 2. Summarization
- Retrieved documents are summarized using the fine-tuned LLM.
- The prompt format ensures the model knows where to start the response:
  ```text
  ### Instruction: Summarize the following legal text.

  ### Input:
  {retrieved_document}

  ### Response:
  {generated_summary}
  ```

---

## Running the Project

### 1. Fine-Tuning
Run the fine-tuning script:
```bash
python src/fine_tune.py
```

### 2. Inference with RAG
Run the RAG pipeline for document retrieval and summarization:
```bash
python src/rag_pipeline.py
```

---

## Future Work

1. **Increase Token Limit**: The current model supports up to 4096 tokens. Future work can explore extending this limit for longer documents.
2. **Expand to UK Dataset**: Adapt the model for summarizing UK legal documents, which are typically larger and more complex.
3. **Optimize Retrieval**: Improve the RAG pipeline for faster and more accurate document retrieval.

---

## References

1. [Llama 2 Documentation](https://huggingface.co/docs/transformers/model_doc/llama2)
2. [LoRA Fine-Tuning with AMD ROCm](https://rocm.blogs.amd.com/artificial-intelligence/llama2-lora/README.html)
3. [SFTTrainer Documentation](https://huggingface.co/docs/trl/en/sft_trainer)
4. [4-bit Quantization with Bitsandbytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
5. [Fine-Tuning LLMs with Domain Knowledge](https://github.com/aws-samples/fine-tuning-llm-with-domain-knowledge)

---

This project provides a robust framework for fine-tuning LLMs for legal document summarization and integrating them into a RAG pipeline for efficient retrieval and generation.