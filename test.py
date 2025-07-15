"""
Test script for ExMed-BERT Patient class functionality.
This script demonstrates how to create and process patient data using the ExMed-BERT framework.
It includes example data creation, encoding, and visualization of the results.

# This script is designed to show how to prepare patient data for pretraining ExMed-BERT.
# It covers the creation of encoding dictionaries, patient data, and dataset objects,
# and demonstrates how to encode, mask, and save data for model pretraining.
"""

import sys
import os
from datetime import date

# Add the ExMed-BERT package to Python path for imports
sys.path.append('./ExMed-BERT-main')  # Ensures local package is importable

import torch  # PyTorch is used for tensor operations and model input formatting

# Import necessary encoding classes for different data types
from exmed_bert.data.encoding import (
    AgeDict,     # Handles age binning and encoding
    CodeDict,    # Handles medical codes (ICD, PheWAS, RxNorm, ATC)
    SexDict,     # Handles sex/gender encoding
    StateDict,   # Handles US state encoding
    DICT_DEFAULTS,  # Default tokens (e.g., PAD, UNK, MASK)
    EndpointDict # Handles endpoint label encoding (e.g., for classification tasks)
)
from exmed_bert.data.patient import Patient  # Main class for patient sequence processing
from exmed_bert.data.dataset import PatientDataset  # Dataset class for batching patients

# --- Expanded code dictionaries and mappings for longer sequences ---
# These are toy/miniature examples. In real use, these would be much larger.

# ATC (Anatomical Therapeutic Chemical) codes for medications (expanded)
atc_codes = [
    "A01AA01", "B01AC06", "C09AA05", "D05AX02", "E03AA01", "F01BA01", "G04BE03"
]

# PheWAS (Phenome Wide Association Study) codes for diagnoses (expanded)
phewas_codes = [
    "008", "250", "401.1", "530.11", "715.2", "272.1", "585.3"
]

# Mapping from RxNorm (prescription) to ATC codes (expanded)
rx_to_atc_map = {
    "860975": "A01AA01",
    "197361": "B01AC06",
    "123456": "C09AA05",
    "654321": "D05AX02",
    "789012": "E03AA01",
    "345678": "F01BA01",
    "987654": "G04BE03"
}

# Mapping from ICD-10 to PheWAS codes (expanded)
icd_to_phewas_map = {
    "I10": "401.1",
    "E11.9": "250",
    "Z51.11": "008",
    "K21.0": "530.11",
    "M17.9": "715.2",
    "E78.5": "272.1",
    "N18.3": "585.3"
}

# List of US states for state encoding
state_list = ["CA", "NY", "TX"]

# Initialize encoding dictionaries for different data types
# These objects map raw data (codes, ages, etc.) to integer IDs for model input
code_dict = CodeDict(
    atc_codes=atc_codes,                # List of valid ATC codes for drugs
    phewas_codes=phewas_codes,          # List of valid PheWAS codes for diagnoses
    rx_to_atc_map=rx_to_atc_map,        # RxNorm to ATC mapping for drug normalization
    icd_to_phewas_map=icd_to_phewas_map # ICD to PheWAS mapping for diagnosis normalization
)
# The following attributes provide access to the code vocabulary and mappings
code_dict.vocab  # List of all tokens (codes + special tokens)
code_dict.entities  # List of entity types (e.g., atc, phewas)
code_dict.entity_to_id  # Mapping from entity type to integer ID
code_dict.ids_to_entity  # Reverse mapping
code_dict.labels_to_id  # Mapping from code label to integer ID
code_dict.ids_to_label  # Reverse mapping

code_dict.atc_codes  # List of ATC codes
code_dict.phewas_codes  # List of PheWAS codes
code_dict.rx_atc_map  # RxNorm to ATC mapping
code_dict.icd_phewas_map  # ICD to PheWAS mapping

# Age dictionary for binning and encoding patient ages - modified for yearly ages
age_dict = AgeDict(
    max_age=90,    # Maximum age to consider
    min_age=0,     # Minimum age to consider (changed to 0 to include infants)
    binsize=1      # Size of age bins (1 year)
)
# Convert the age_dict's vocabulary to integers for consistency
age_dict.vocab = [str(int(float(age))) if age not in DICT_DEFAULTS else age for age in age_dict.vocab]
age_dict.labels_to_id = {(str(int(float(label))) if label not in DICT_DEFAULTS else label): idx 
                        for label, idx in age_dict.labels_to_id.items()}
age_dict.ids_to_label = {idx: (str(int(float(label))) if label not in DICT_DEFAULTS else label)
                        for idx, label in age_dict.ids_to_label.items()}

age_dict.entities  # List of age entity types
age_dict.entity_to_id  # Mapping from entity type to ID
age_dict.ids_to_entity  # Reverse mapping
age_dict.labels_to_id  # Mapping from label to ID
age_dict.ids_to_label  # Reverse mapping

# Sex dictionary for encoding patient sex/gender
sex_dict = SexDict(sex=["MALE", "FEMALE"])
sex_dict.vocab  # List of all sex tokens
sex_dict.entities  # List of entity types
sex_dict.entity_to_id  # Mapping from entity type to ID
sex_dict.ids_to_entity  # Reverse mapping
sex_dict.labels_to_id  # Mapping from label to ID
sex_dict.ids_to_label  # Reverse mapping

# State dictionary for encoding US states
state_dict = StateDict(states=state_list)
state_dict.vocab  # List of all state tokens
state_dict.entities  # List of entity types
state_dict.entity_to_id  # Mapping from entity type to ID
state_dict.ids_to_entity  # Reverse mapping
state_dict.labels_to_id  # Mapping from label to ID
state_dict.ids_to_label  # Reverse mapping

# Example patient data with longer medical histories (expanded sequences)
# Each patient is a dictionary with all required fields for Patient class
patients = [
    {
        # First patient example (expanded)
        "patient_id": 12345,  # Unique patient identifier
        "diagnoses": [
            "I10", "E11.9", "Z51.11", "K21.0", "M17.9", "E78.5", "N18.3"
        ],  # ICD-10 codes (7 codes)
        "diagnosis_dates": [
            date(2020, 3, 15),
            date(2020, 3, 15),
            date(2020, 4, 20),
            date(2020, 5, 1),
            date(2020, 5, 10),
            date(2020, 6, 5),
            date(2020, 7, 12)
        ],
        "drugs": [
            "860975", "197361", "197361", "654321", "789012", "345678", "987654"
        ],  # RxNorm codes (7 codes)
        "prescription_dates": [
            date(2020, 3, 16),
            date(2020, 3, 16),
            date(2020, 4, 21),
            date(2020, 5, 2),
            date(2020, 5, 11),
            date(2020, 6, 6),
            date(2020, 7, 13)
        ],
        "birth_year": 2004,  # Year of birth
        "sex": "MALE",      # Sex/gender
        "patient_state": "CA"  # US state
    },
    {
        # Second patient example (expanded)
        "patient_id": 67890,
        "diagnoses": [
            "E11.9", "I10", "K21.0", "M17.9", "E78.5", "N18.3", "Z51.11"
        ],  # ICD-10 codes (7 codes)
        "diagnosis_dates": [
            date(2020, 5, 1),
            date(2020, 5, 15),
            date(2020, 5, 20),
            date(2020, 6, 1),
            date(2020, 6, 10),
            date(2020, 7, 5),
            date(2020, 8, 12)
        ],
        "drugs": [
            "123456", "860975", "654321", "789012", "345678", "987654", "197361"
        ],  # RxNorm codes (7 codes)
        "prescription_dates": [
            date(2020, 5, 2),
            date(2020, 5, 16),
            date(2020, 5, 21),
            date(2020, 6, 2),
            date(2020, 6, 11),
            date(2020, 7, 6),
            date(2020, 8, 13)
        ],
        "birth_year": 1960,
        "sex": "FEMALE",
        "patient_state": "NY"
    }
]


# Process each patient and demonstrate the encoding/decoding functionality
for idx, patient_data in enumerate(patients, 1):
    try:
        # Create Patient instance with configuration parameters
        # (All parameters are shown for clarity and customization)
        patient = Patient(
            patient_id=patient_data["patient_id"],
            diagnoses=patient_data["diagnoses"],
            drugs=patient_data["drugs"],
            diagnosis_dates=patient_data["diagnosis_dates"],
            prescription_dates=patient_data["prescription_dates"],
            birth_year=patient_data["birth_year"],
            sex=patient_data["sex"],
            patient_state=patient_data["patient_state"],
            max_length=50,
            code_embed=code_dict,
            sex_embed=sex_dict,
            age_embed=age_dict,
            state_embed=state_dict,
            mask_drugs=True,
            delete_temporary_variables=True,
            split_sequence=True,
            drop_duplicates=True,
            converted_codes=False,
            convert_icd_to_phewas=True,
            convert_rxcui_to_atc=True,
            keep_min_unmasked=1,
            max_masked_tokens=20,
            masked_lm_prob=0.15,
            truncate="right",
            index_date=None,
            had_plos=None,
            endpoint_labels=patient_data.get("endpoint_labels", None),
            dynamic_masking=False,
            min_observations=5,
            age_usage="year",
            use_cls=True,
            use_sep=True,
            valid_patient=True,
            num_visits=None,
            combined_length=None,
            unpadded_length=None
        )
        print(f'Valid Patient: {patient.valid_patient}')
        # Debug: Print drugs, prescription_dates, and combined codes
        print(f"\n[DEBUG] Patient {idx} drugs: {patient.drugs}")
        print(f"[DEBUG] Patient {idx} prescription_dates: {patient.prescription_dates}")
        print(f"[DEBUG] Patient {idx} combined codes: {getattr(patient, 'codes', None)}")
        print(f"[DEBUG] Patient {idx} birth_year: {patient.birth_year}")
        print(f"[DEBUG] Patient {idx} age_ids: {getattr(patient, 'age_ids', None)}")
        
        # Get encoded data for model input
        model_input = patient.get_patient_data(
            evaluate=True,           # Get evaluation format (MLM labels)
            mask_dynamically=False,   # Use static masking
            min_unmasked=1,         # Minimum unmasked tokens
            max_masked=20,          # Maximum masked tokens
            masked_lm_prob=0.15,    # Masking probability
            mask_drugs=True         # Mask drug codes as well
        )
        
        # Display encoded model input
        print(f"\nPatient {idx} encoded model_input:")
        for k, v in model_input.items():
            print(f"  {k}: {v}")
            
        # Display decoded model input (human-readable format)
        print(f"\nPatient {idx} decoded model_input:")
        
        # Decode and display each type of encoded data
        if "input_ids" in model_input:
            print("  input_ids (codes):", code_dict.decode(model_input["input_ids"]))
        
        if "sex_ids" in model_input:
            print("  sex_ids:", sex_dict.decode(model_input["sex_ids"]))
        
        if "state_ids" in model_input:
            print("  state_ids:", state_dict.decode(model_input["state_ids"]))
        
        if "age_ids" in model_input:
            print("  age_ids:", age_dict.decode(model_input["age_ids"]))
        
        if "entity_ids" in model_input:
            # Convert entity IDs to human-readable labels
            print("  entity_ids:", [code_dict.ids_to_entity.get(x.item(), "UNK") for x in model_input["entity_ids"]])
        
        if "code_labels" in model_input:
            print("  code_labels:", code_dict.decode(model_input["code_labels"]))
            
        # Convert to DataFrame for easier visualization
        print(f"\nPatient {idx} as DataFrame:")
        df = patient.to_df(
            # Pass all necessary encoding dictionaries
            code_embed=code_dict,
            age_embed=age_dict,
            sex_embed=sex_dict,
            state_embed=state_dict,
            
            # Masking configuration
            dynamic_masking=True,
            mask_drugs=True,
            min_unmasked=1,
            max_masked=20,
            masked_lm_prob=0.15
        )
        print(df)
        
    except Exception as e:
        print(f"Error processing patient {idx}: {e}")



############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################


# --- Additional code to create and save a PatientDataset for pretraining ---

# 1. Define endpoint labels for pretraining (commonly 'plos' for Prolonged Length of Stay)
endpoint_labels = ["plos"]  # List of endpoint names for classification
endpoint_dict = EndpointDict(endpoint_labels)  # Encodes endpoint labels to IDs

# 2. Add a 'plos' label to each patient (for demo, alternate 0/1)
for i, patient in enumerate(patients):
    # The endpoint_labels must be a torch.LongTensor of length = len(endpoint_labels)
    # Here, we alternate 0 and 1 for demonstration
    patient["endpoint_labels"] = torch.LongTensor([i % 2])

# 3. Create Patient objects with endpoint_labels
#    Each Patient object represents a single patient's medical history, encoded for model input
patient_objs = []
for patient_data in patients:
    patient_obj = Patient(
        patient_id=patient_data["patient_id"],  # Unique patient ID
        diagnoses=patient_data["diagnoses"],    # List of diagnosis codes
        drugs=patient_data["drugs"],            # List of drug codes
        diagnosis_dates=patient_data["diagnosis_dates"],  # Dates for diagnoses
        prescription_dates=patient_data["prescription_dates"],  # Dates for prescriptions
        birth_year=patient_data["birth_year"],  # Year of birth
        sex=patient_data["sex"],                # Sex/gender
        patient_state=patient_data["patient_state"],  # US state
        max_length=16,           # Maximum sequence length for model input
        code_embed=code_dict,    # CodeDict for encoding medical codes
        sex_embed=sex_dict,      # SexDict for encoding sex/gender
        age_embed=age_dict,      # AgeDict for encoding age
        state_embed=state_dict,  # StateDict for encoding state
        mask_drugs=True,         # Whether to mask drug codes for MLM
        delete_temporary_variables=True,  # Clean up after processing
        split_sequence=True,     # Whether to split long sequences
        drop_duplicates=True,    # Remove duplicate codes
        converted_codes=False,   # Whether codes are already converted
        convert_icd_to_phewas=True,  # Convert ICD to PheWAS
        convert_rxcui_to_atc=True,   # Convert RxNorm to ATC
        keep_min_unmasked=1,     # Minimum unmasked tokens for MLM
        max_masked_tokens=20,    # Maximum masked tokens for MLM
        masked_lm_prob=0.15,     # Probability of masking each token
        truncate="right",       # Truncate long sequences from the right
        index_date=None,         # Optional index date (not used here)
        had_plos=None,           # Optional binary outcome (not used here)
        endpoint_labels=patient_data.get("endpoint_labels", None),  # Endpoint label(s)
        dynamic_masking=False,   # Use static masking for pretraining
        min_observations=5,      # Minimum number of observations required
        age_usage="year",       # Use age in years
        use_cls=True,            # Add CLS token at start
        use_sep=True,            # Add SEP token between visits
        valid_patient=True,      # Assume valid for demo
        num_visits=None,         # Will be set during processing
        combined_length=None,    # Will be set during processing
        unpadded_length=None     # Will be set during processing
    )
    patient_objs.append(patient_obj)

# 4. Create the PatientDataset
#    This object batches multiple Patient objects and prepares them for model training
patient_dataset = PatientDataset(
    code_embed=code_dict,      # CodeDict for encoding codes
    age_embed=age_dict,        # AgeDict for encoding age
    sex_embed=sex_dict,        # SexDict for encoding sex/gender
    state_embed=state_dict,    # StateDict for encoding state
    endpoint_dict=endpoint_dict,  # EndpointDict for endpoint labels
    patient_paths=None,        # Not using file paths (RAM only)
    max_length=16,             # Maximum sequence length
    do_eval=True,              # Pretraining mode (MLM)
    mask_substances=True,      # Mask both diagnoses and drugs
    dataset_path=None,         # Not saving to disk yet
    patients=patient_objs,     # List of Patient objects
    dynamic_masking=False,     # Use static masking
    min_unmasked=1,           # Minimum unmasked tokens
    max_masked=20,            # Maximum masked tokens
    masked_lm_prob=0.15       # Masking probability
)

# 5. Save the dataset to disk for use in pretraining
#    This will create a .pt file and a directory with joblib patient files
output_path = "./demo_patient_dataset.pt"  # Output file for the dataset
patient_dataset.save_dataset(output_path)   # Save dataset for later use
print(f"\nPatientDataset saved to {output_path}. You can now use this file for pretraining.")

# --- End of PatientDataset creation and saving code ---


