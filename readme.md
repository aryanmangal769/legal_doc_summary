conda create --name legal_assitant python=3.10
conda activate legal_assitant
pip install -r requirements.txt

Download Llama 2 from thsi website
https://www.llama.com/llama-downloads/

pip install llama-stack
lama download --source meta --model-id Llama-2-7b  (YOu wull need to give the custom email generated from their website)'


Fill the form here : 
https://huggingface.co/meta-llama/Llama-2-7b-hf

THen you will get acces in your hugging face.
huggingface-cli login  # Ensure you're logged in with an authorized account

from transformers import LlamaForCausalLM, LlamaTokenizer
model_name = "meta-llama/Llama-2-7b"
model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=True)

pip install sentencepiece

#tried some datasets to download that have a predecided folder structure but this looks the best to download as of now.
https://zenodo.org/records/7152317#.ZCSfaoTMI2y

Download the dataset.zip from here. 

Afger downoad the data in legal-llm-project/datasets

Do the preprocsiing: 
python src/data_preprocessing.py 

pip install datasets
pip install trl
pip install bitsandbytes




Referneces: 
https://rocm.blogs.amd.com/artificial-intelligence/llama2-lora/README.html#
https://discuss.huggingface.co/t/llama-7b-gpu-memory-requirement/34323/8


https://huggingface.co/docs/trl/en/sft_trainer  Very good documentation on SFT trainer.

Normal 7b model does not work . We needed to qualtize it to 4bit to get it working: https://huggingface.co/blog/4bit-transformers-bitsandbytes



