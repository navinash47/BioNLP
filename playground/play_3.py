#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# %%
import os
model_base = "/project/pi_hongyu_umass_edu/shared_llm_checkpoints/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B"
snapshots_dir = os.path.join(model_base, "snapshots")
snapshot = os.listdir(snapshots_dir)[0]  # Get the first (and probably only) snapshot
model_path = os.path.join(snapshots_dir, snapshot)



# %%
print(f"Loading from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                           torch_dtype=torch.float16, 
                                           device_map="auto",
                                           trust_remote_code=True)
# %%
print("Current CUDA device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Method 2: Check model's device
print("Model device:", next(model.parameters()).device)

# Method 3: More detailed GPU info
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_properties(i)}")

# %%
question = "A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee. A culture of the joint fluid shows a bacteria that does not ferment maltose and has no polysaccharide capsule. The physician orders antibiotic therapy for the patient. The mechanism of action of action of the medication given blocks cell wall synthesis, which of the following was given?"
answer = "Ceftriaxone"
options = {"A": "Chloramphenicol", "B": "Gentamicin", "C": "Ciprofloxacin", "D": "Ceftriaxone", "E": "Trimethoprim"}
meta_info = "step1"
answer_idx = "D"

# Format the prompt
prompt = f"""Question: {question}

Options:
A) {options['A']}
B) {options['B']}
C) {options['C']}
D) {options['D']}
E) {options['E']}

Please select the correct answer and explain why."""

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs["input_ids"],
    max_length=2048,
    temperature=0.5,
    do_sample=True,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
    return_dict_in_generate=True,
    output_scores=True
)

# Decode and print the response
response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print("\nModel Response:")
print(response)


# %%
