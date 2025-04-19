from transformers import AutoModelForCausalLM, AutoTokenizer

# For DeepSeek-R1-Distill-Qwen-1.5B
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
