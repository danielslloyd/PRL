"""
Debug script to check position IDs in synthetic data
"""
import torch
import sys

# Add project to path
sys.path.insert(0, '.')

from exmed_bert.data.dataset import PatientDataset

# Load the training dataset
print("Loading training dataset...")
dataset = PatientDataset.load_dataset('pretrain_stuff/synthetic_train.pt')

print(f"Dataset has {len(dataset)} patients")
print(f"Max position embeddings in config: 50")
print()

# Check first few patients for position_ids
print("Checking position_ids for first 10 patients:")
for i in range(min(10, len(dataset))):
    patient_data = dataset[i]
    position_ids = patient_data['position_ids']
    max_pos = position_ids.max().item()
    print(f"Patient {i}: max position_id = {max_pos}, length = {len(position_ids)}")

    if max_pos >= 50:
        print(f"  ⚠️  ERROR: Position ID {max_pos} exceeds max_position_embeddings (50)!")
        print(f"  Full position_ids: {position_ids.tolist()}")
        break

print()
print("Now checking ALL patients for max position_ids...")
max_overall = 0
problem_patients = []

for i in range(len(dataset)):
    patient_data = dataset[i]
    position_ids = patient_data['position_ids']
    max_pos = position_ids.max().item()

    if max_pos > max_overall:
        max_overall = max_pos

    if max_pos >= 50:
        problem_patients.append((i, max_pos))

print(f"Max position_id across all patients: {max_overall}")
print(f"Number of patients with position_id >= 50: {len(problem_patients)}")

if problem_patients:
    print("\nFirst 10 problem patients:")
    for patient_idx, max_pos in problem_patients[:10]:
        print(f"  Patient {patient_idx}: max position_id = {max_pos}")
