import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig  # Import SFTTrainer from trl
from peft import LoraConfig, get_peft_model

# Step 1: Set up directories
preprocessed_data_dir = "../dataset/processed-IN-Ext/"

# Step 2: Load the tokenizer and model
model_name = "meta-llama/Llama-2-7b-hf"  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name, load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token

# Step 3: Set up LoRA configuration
lora_config = LoraConfig(
    lora_alpha=8,          # Scaling factor for low-rank matrices
    lora_dropout=0.1,      # Dropout rate for LoRA layers
    r=8,                   # Rank (size of low-rank matrices)
    bias="none",           # No bias in LoRA layers
    task_type="CAUSAL_LM"  # Task type for causal language modeling (autoregressive)
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Print the number of trainable parameters
model.print_trainable_parameters()  # This shows the amount of LoRA params that will be trained

# Step 4: Load and preprocess the dataset
def load_dataset(jsonl_file):
    """
    Load preprocessed data and format it into a structured text field.
    We clip the judgement at 100 character becuase we cannnout procss ore than 4096 tokens.
    """
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Define a system prompt (instruction)
    system_prompt = "Summarize the following legal text."

    # Format each example with clear distinction
    texts = []
    for item in data:
        text = f"""### Instruction: {system_prompt}

### Input:
{item['judgement'].strip()[:10000]}

### Response:
{item['summary'].strip() }
""".strip()
        texts.append(text)

    # Create a dataset with a single "text" column
    dataset = Dataset.from_dict({"text": texts})
    return dataset



# Load datasets
train_file_A1 = os.path.join(preprocessed_data_dir, "full_summaries_A1.jsonl")
train_file_A2 = os.path.join(preprocessed_data_dir, "full_summaries_A2.jsonl")

train_dataset_A1 = load_dataset(train_file_A1)
train_dataset_A2 = load_dataset(train_file_A2)


# Combine datasets
train_data = concatenate_datasets([train_dataset_A1, train_dataset_A2])

# Step 5: Set up training parameters
train_params = SFTConfig(
    output_dir="../results_lora",         # Output directory for model checkpoints
    num_train_epochs=3,                # Number of epochs
    per_device_train_batch_size=1,      # Batch size per device (adjust based on available memory)
    gradient_accumulation_steps=1,      # Accumulate gradients before updating model
    optim="paged_adamw_32bit",          # Optimizer to use
    save_steps=50,                      # Save checkpoints every 50 steps
    logging_steps=50,                   # Log training progress every 50 steps
    learning_rate=5e-3,                 # Learning rate
    weight_decay=0.001,                 # Weight decay for regularization
    fp16=True,                          # Enable mixed precision for stability
    bf16=False,                         # Disable bfloat16
    max_grad_norm=0.3,                  # Gradient clipping norm
    warmup_ratio=0.03,                  # Warm-up ratio for learning rate scheduler
    group_by_length=True,               # Group samples by length to minimize padding
    lr_scheduler_type="constant",       # Use a constant learning rate
    report_to="tensorboard",            # Log to TensorBoard for visualization
    dataset_text_field="text",          # Column containing the text for training
    max_seq_length=4096                 # Maximum sequence length for input text
)

# Step 6: Initialize Trainer with LoRA model
fine_tuning = SFTTrainer(
    model=model,
    train_dataset=train_data,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=train_params
)

# Step 7: Start fine-tuning the model
print("Starting fine-tuning...")
fine_tuning.train()

# Step 8: Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("../fine_tuned_lora_model")
tokenizer.save_pretrained("../fine_tuned_lora_model")
print("Fine-tuned model saved at '../fine_tuned_lora_model'")