import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig # Import SFTTrainer from trl
from peft import LoraConfig, get_peft_model


preprocessed_data_dir = "../dataset/processed-IN-Ext/"

# Step 1: Load the tokenizer and model
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name,load_in_8bit=True )
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token

# Step 2: Set up LoRA configuration
lora_config = LoraConfig(
    lora_alpha=8,          # scaling factor for low-rank matrices
    lora_dropout=0.1,      # dropout rate for LoRA layers
    r=8,                   # rank (size of low-rank matrices)
    bias="none",           # no bias in LoRA layers
    task_type="CAUSAL_LM"  # task type for causal language modeling (autoregressive)
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print the number of trainable parameters
model.print_trainable_parameters()  # This shows the amount of LoRA params that will be trained

# Step 3: Load your preprocessed dataset
def load_dataset(jsonl_file, tokenizer, max_length=512):
    """
    Load preprocessed data and tokenize it.
    """
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    inputs = [
        f"Summarize the following legal text: {item['judgement']}"
        for item in data
    ]
    outputs = [item["summary"] for item in data]

    # Tokenization function
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,  # Ensure attention mask is included
        )
        labels = tokenizer(
            examples["output_text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    dataset = Dataset.from_dict({"input_text": inputs, "output_text": outputs})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "output_text"])
    return tokenized_dataset


train_file_A1 = os.path.join(preprocessed_data_dir, "full_summaries_A1.jsonl")
train_file_A2 = os.path.join(preprocessed_data_dir, "full_summaries_A2.jsonl")

# Load datasets
train_dataset_A1 = load_dataset(train_file_A1, tokenizer, max_length=2048)
train_dataset_A2 = load_dataset(train_file_A2, tokenizer, max_length=2048)

# Combine datasets
train_data = concatenate_datasets([train_dataset_A1, train_dataset_A2])

# Step 4: Set up training parameters
train_params = SFTConfig(
    output_dir="../results_lora",         # Output directory for model checkpoints
    num_train_epochs=10,                 # Number of epochs
    per_device_train_batch_size=1,      # Batch size per device (adjust based on available memory)
    gradient_accumulation_steps=1,      # Accumulate gradients before updating model
    optim="paged_adamw_32bit",          # Optimizer to use     # Cannot use adam bevuse it requires 4x more memory : 
    save_steps=50,                      # Save checkpoints every 50 steps
    logging_steps=50,                   # Log training progress every 50 steps
    learning_rate=1e-3,                 # Learning rate
    weight_decay=0.001,                 # Weight decay for regularization
    fp16=True,                         # Disable mixed precision for stability (enable if you want FP16)
    bf16=False,                         # Disable bfloat16
    max_grad_norm=0.3,                  # Gradient clipping norm
    warmup_ratio=0.03,                  # Warm-up ratio for learning rate scheduler
    group_by_length=True,               # Group samples by length to minimize padding
    lr_scheduler_type="constant",      # Use a constant learning rate
    report_to="tensorboard",               # Log to TensorBoard for visualization
    dataset_text_field="labels",          
)

# Step 5: Initialize Trainer with LoRA model
fine_tuning = SFTTrainer(
    model=model,
    train_dataset=train_data,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=train_params
)

# Step 6: Start fine-tuning the model
print("Starting fine-tuning...")
fine_tuning.train()

# Step 7: Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("../fine_tuned_lora_model")
tokenizer.save_pretrained("../fine_tuned_lora_model")
print("Fine-tuned model saved at '../fine_tuned_lora_model'")
