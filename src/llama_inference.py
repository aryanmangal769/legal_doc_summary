from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the model and tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # Model name for Llama-2 7B

model_name = "../fine_tuned_lora_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# tokenizer = AutoTokenizer.from_pretrained(model_name,load_in_8bit=True )
# model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your input prompt
path = "/data/aryan/extras/LLM_project/legal-llm-project/dataset/IN-Ext/judgement/1953_L_1.txt"
text = open(path, "r").read()
input_text = "Summarize the following legal text: " + text

# input_text = "What is life?"


# Tokenize the input text with truncation explicitly set to True and move to the same device
inputs = tokenizer(input_text, max_length=2048, truncation=True, return_tensors="pt").to(model.device)

# Perform inference (generate text from the input)
generated_output = model.generate(inputs["input_ids"], max_new_tokens=200)

# Decode and print the generated output
# generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
generated_text = tokenizer.decode(generated_output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

# print(len(generated_text))
print("Generated Output:")
print(generated_text)