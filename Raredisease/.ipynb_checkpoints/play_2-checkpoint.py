# %%
from vllm import LLM, SamplingParams
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import re
model_base = "/project/pi_hongyu_umass_edu/shared_llm_checkpoints/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
snapshots_dir = os.path.join(model_base, "snapshots")
snapshot = os.listdir(snapshots_dir)[0]  # Get the first (and probably only) snapshot
model_path = os.path.join(snapshots_dir, snapshot)

#%%
import torch
def monitor_gpu_memory(message=""):
    """Monitor GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"\nGPU Memory ({message}):")
    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved:  {reserved:.2f} MB")


#%%
total_memory = torch.cuda.get_device_properties(0).total_memory
allocated_memory = torch.cuda.memory_allocated()
free_memory = total_memory - allocated_memory
print(f"Total GPU Memory: {total_memory / 1024**2:.2f} MB")
print(f"Allocated Memory: {allocated_memory / 1024**2:.2f} MB")
print(f"Free Memory: {free_memory / 1024**2:.2f} MB")   

#%%
# llm = LLM(
#     model=model_path,
#     trust_remote_code=True,
#     dtype="float16",
#     # gpu_memory_utilization=0.50,  # Reduced from 0.95 to 0.80
#     tensor_parallel_size=1,
#     max_num_batched_tokens=2048,  # Reduced from 4096
#     max_num_seqs=64,  # Reduced from 256
#     # enforce_eager=True,  # Add this to ensure eager execution
#     max_model_len=1024,
# )

# %%
print(f"Loading from: {model_path}")
monitor_gpu_memory("Before initialization")
# Initialize vLLM engine
# Calculate available GPU memory
total_memory = torch.cuda.get_device_properties(0).total_memory
allocated_memory = torch.cuda.memory_allocated()
free_memory = total_memory - allocated_memory

llm = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype="float16",
    gpu_memory_utilization=0.6,  # Reduced from 0.8
    tensor_parallel_size=1,
    max_num_batched_tokens=1024,  # Reduced from 2048
    max_num_seqs=32,  # Reduced from 64
    max_model_len=512,  # Reduced from 1024
)

monitor_gpu_memory("After initialization")
#%%
# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    # use_beam_search=False,
)

# %%
core_files = [
    'history_of_present_illness.csv',    # HPI narratives
    'physical_examination.csv',          # PE findings
    'laboratory_tests.csv',              # 138,788 lab results
    'radiology_reports.csv',             # 5,959 imaging reports
    'discharge_diagnosis.csv'            # Ground truth labels
]
# %%
def format_clinical_prompt(patient_data):
    """Structure multimodal clinical data"""
    return f"""Clinical Decision Task:
Patient History: {patient_data['hpi']}
Physical Exam Findings: {patient_data['pe']}
Laboratory Results: {patient_data['labs']}
Imaging Reports: {patient_data['imaging']}

Generate differential diagnosis and recommended treatment steps:"""
# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, classification_report
import re



#%%
def extract_diagnosis_from_response(response):
    """Extract the primary diagnosis from the LLM response"""
    # Look for diagnosis after common markers
    markers = ["primary diagnosis:", "diagnosis:", "differential diagnosis:", "most likely diagnosis:"]
    for marker in markers:
        if marker.lower() in response.lower():
            # Extract text after the marker
            start_idx = response.lower().find(marker.lower()) + len(marker)
            diagnosis = response[start_idx:].split('\n')[0].strip()
            return diagnosis
    return response.split('\n')[0].strip()  # Return first line if no marker found

def evaluate_predictions(predictions, ground_truth):
    """Calculate accuracy and generate classification report"""
    # Convert predictions and ground truth to lists if they aren't already
    pred_list = predictions if isinstance(predictions, list) else predictions.tolist()
    gt_list = ground_truth if isinstance(ground_truth, list) else ground_truth.tolist()
    
    # Calculate accuracy
    accuracy = accuracy_score(gt_list, pred_list)
    
    # Generate classification report
    report = classification_report(gt_list, pred_list)
    
    return accuracy, report

#%%
# data_dir = "/project/pi_hongyu_umass_edu/shared_datasets/physionet.org/files/mimic-iv-ext-cdm/1.1"
# discharge_df = pd.read_csv(os.path.join(data_dir, 'discharge_diagnosis.csv'))

#%%
# predictions = []
# ground_truth = []

# test_size = 10  # Adjust this number based on your needs
# test_patients = discharge_df['hadm_id'].unique()[:test_size]
# print(f'test_patients: {test_patients}')
def load_patient_data(hadm_id, data_dir):
    """Load all clinical data for a specific patient"""
    patient_data = {}
    
    # Load HPI
    hpi_df = pd.read_csv(os.path.join(data_dir, 'history_of_present_illness.csv'))
    patient_data['hpi'] = hpi_df[hpi_df['hadm_id'] == hadm_id]['hpi'].iloc[0] if not hpi_df[hpi_df['hadm_id'] == hadm_id].empty else "No HPI available"
    
    # Load PE
    pe_df = pd.read_csv(os.path.join(data_dir, 'physical_examination.csv'))
    patient_data['pe'] = pe_df[pe_df['hadm_id'] == hadm_id]['pe'].iloc[0] if not pe_df[pe_df['hadm_id'] == hadm_id].empty else "No PE findings available"
    
    # Load Labs
    labs_df = pd.read_csv(os.path.join(data_dir, 'laboratory_tests.csv'))
    patient_labs = labs_df[labs_df['hadm_id'] == hadm_id]
    patient_data['labs'] = "\n".join(patient_labs.apply(lambda x: f"{x['itemid']}: {x['valuestr']}", axis=1)) if not patient_labs.empty else "No lab results available"
    # Load Imaging
    imaging_df = pd.read_csv(os.path.join(data_dir, 'radiology_reports.csv'))
    patient_imaging = imaging_df[imaging_df['hadm_id'] == hadm_id]
    patient_data['imaging'] = "\n".join(patient_imaging.apply(lambda x: f"{x['modality']} {x['region']} - {x['exam_name']}:\n{x['text']}", axis=1)) if not patient_imaging.empty else "No imaging reports available"
    return patient_data




#%%
def main():
    data_dir = "/project/pi_hongyu_umass_edu/shared_datasets/physionet.org/files/mimic-iv-ext-cdm/1.1"
    
    # Load ground truth diagnoses
    discharge_df = pd.read_csv(os.path.join(data_dir, 'discharge_diagnosis.csv'))
    
    # Initialize lists to store predictions and ground truth
    predictions = []
    ground_truth = []
    
    # Process a subset of patients for testing
    test_size = 10  # Adjust this number based on your needs
    test_patients = discharge_df['hadm_id'].unique()[:test_size]
    
    print(f"Processing {test_size} patients...")
    
    # Prepare all prompts first for batch processing
    prompts = []
    for hadm_id in test_patients:
        # Get ground truth diagnosis
        patient_diagnosis = discharge_df[discharge_df['hadm_id'] == hadm_id]['discharge_diagnosis'].iloc[0]
        ground_truth.append(patient_diagnosis)
        
        # Load patient data
        patient_data = load_patient_data(hadm_id, data_dir)
        
        # Generate prompt
        prompt = format_clinical_prompt(patient_data)
        prompts.append(prompt)
    
    # Generate responses in batch
    outputs = llm.generate(prompts, sampling_params)
    
    # Process outputs
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text
        predicted_diagnosis = extract_diagnosis_from_response(response)
        predictions.append(predicted_diagnosis)
        
        # Print results for this patient
        print(f"\nPatient {test_patients[idx]}:")
        print(f"Ground Truth: {ground_truth[idx]}")
        print(f"Predicted: {predicted_diagnosis}")
        print("-" * 80)
    
    # Evaluate results
    accuracy, report = evaluate_predictions(predictions, ground_truth)
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save results to file
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    with open('inference_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
