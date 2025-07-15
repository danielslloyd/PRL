import torch
import os

input_path = 'demo_patient_dataset.pt/patients/12345_0.joblib'
loaded_dataset = torch.load(input_path)
print("Successfully loaded the dataset!")