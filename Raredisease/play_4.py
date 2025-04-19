#%%
from unsloth import FastLanguageModel
import pandas as pd
from datasets import Dataset
from transformers import pipeline

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    token="your_hf_token",
)

# Check the type of the model
print(f"Model type: {type(model)}")

#%%
def format_mimic_data(patient_case):
    return f"""
    ### Patient History:
    {patient_case['hpi']}
    
    ### Physical Examination:
    {patient_case['pe']}
    
    ### Lab Results:
    {patient_case['laboratory_tests']}
    
    ### Imaging Findings:
    {patient_case['radiology_reports']}
    
    ### Diagnosis:
    {patient_case['discharge_diagnosis']}
    """

#%%
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)

#%%
from trl import SFTTrainer
from transformers import TrainingArguments

path = '/Users/jiajun/Desktop/Raredisease/mimic_data'
# Load CSV files with specified column names and data types
hpi_df = pd.read_csv(path + '/history_of_present_illness.csv')
pe_df = pd.read_csv(path + '/physical_examination.csv')

# Specify data types for each column
dtype_spec = {
    'hadm_id': str,
    'itemid': str,
    'valuestr': str,
    'ref_range_lower': str,
    'ref_range_upper': str
}

# Load the CSV file with specified column names and data types
lab_df = pd.read_csv(
    path + '/laboratory_tests.csv',
    names=['hadm_id', 'itemid', 'valuestr', 'ref_range_lower', 'ref_range_upper'],
    dtype=dtype_spec,
    low_memory=False
)
radiology_df = pd.read_csv(path + '/radiology_reports.csv', names=['hadm_id', 'note_id', 'modality', 'region', 'exam_name', 'text'])
diagnosis_df = pd.read_csv(path + '/discharge_diagnosis.csv')

# Fill missing values
lab_df.fillna('N/A', inplace=True)
radiology_df.fillna('N/A', inplace=True)

# Preprocess and format data
def preprocess_data():
    formatted_cases = []
    for index, row in hpi_df.iterrows():
        patient_case = {
            'hpi': row['hpi'],
            'pe': pe_df.loc[index, 'pe'],
            'laboratory_tests': lab_df.loc[index, 'valuestr'],
            'radiology_reports': radiology_df.loc[index, 'text'],
            'discharge_diagnosis': diagnosis_df.loc[index, 'discharge_diagnosis']
        }
        formatted_cases.append(format_mimic_data(patient_case))
    return formatted_cases

formatted_cases = preprocess_data()

# Create a dataset
dataset = Dataset.from_dict({'text': formatted_cases})

#%%
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=300,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        output_dir="outputs",
    ),
    dataset_text_field="text",
)
trainer.train()

#%%
# Save the model and tokenizer
model.save_pretrained("outputs/fine_tuned_model")
tokenizer.save_pretrained("outputs/fine_tuned_model")

# Load the fine-tuned model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained("outputs/fine_tuned_model")

# Check the type of the model
print(f"Model type: {type(model)}")

# Create a pipeline for text generation
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Extract and format data from CSVs for evaluation
# Use the first row as an example
def format_clinical_prompt(patient_case):
    return f"""
    ### Patient History:
    {patient_case['hpi']}
    
    ### Physical Examination:
    {patient_case['pe']}

    ### Lab Results:
    {patient_case['labs']}
    
    ### Imaging Findings:
    {patient_case['imaging']}

    ### Diagnosis:
    
   """

example_index = 0
example_patient_case = {
    'hpi': hpi_df.loc[example_index, 'hpi'],
    'pe': pe_df.loc[example_index, 'pe'],
    'labs': lab_df.loc[example_index, 'valuestr'],  # Changed key to match format_clinical_prompt
    'imaging': radiology_df.loc[example_index, 'text'],  # Changed key to match format_clinical_prompt
}

# Create prompt using the formatting function
clinical_prompt = format_clinical_prompt(example_patient_case)
evaluation_data = [clinical_prompt]

# Generate predictions
for prompt in evaluation_data:
    prediction = text_generator(prompt, max_length=100, num_return_sequences=1)
    print("Input:", prompt)
    print("Prediction:", prediction[0]['generated_text'])
    print("\n")


#%%
#do inference task