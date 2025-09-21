#!/usr/bin/env python3
"""
Regenerate training data with correct ICD entity types
Based on the logic from explain.ipynb
"""

import sys
import os
import yaml
from datetime import date
import numpy as np
from sklearn.model_selection import train_test_split

# Add the project to path
sys.path.append('.')

def main():
    """Regenerate training datasets with ICD entity types"""
    try:
        # Import required classes
        from exmed_bert.data.encoding import (
            AgeDict, CodeDict, SexDict, StateDict, DICT_DEFAULTS
        )
        from exmed_bert.data.patient import Patient
        from exmed_bert.data.dataset import PatientDataset

        print("✓ Successfully imported all required classes")
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

    # Load configuration
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        params = config['data_prep_params_example']
        print(f"✓ Configuration loaded")
        print(f"  convert_icd_to_phewas: {params.get('convert_icd_to_phewas')}")
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

    # Define codes and mappings (from explain.ipynb)
    atc_codes = [
        'A01AA01', 'B01AC06', 'C09AA05', 'D05AX02', 'E03AA01', 'F01BA01', 'G04BE03',
        'H02AB02', 'J01CA04', 'K01AA02', 'L01XE01', 'M01AE01', 'N02BA01', 'O01AA01',
        'P01AB01', 'Q01AA01', 'R03BA02', 'S01AA01', 'T01AA01', 'U01AA01', 'V01AA01'
    ]

    icd_codes = [
        'I10', 'E11.9', 'Z51.11', 'K21.0', 'M17.9', 'E78.5', 'N18.3',
        'A00', 'B00', 'C00', 'D00', 'E00', 'F00', 'G00', 'H00',
        'J00', 'K00', 'L00', 'M00', 'N00'
    ]

    rx_to_atc_map = {
        '860975': 'A01AA01', '197361': 'B01AC06', '123456': 'C09AA05', '654321': 'D05AX02',
        '789012': 'E03AA01', '345678': 'F01BA01', '987654': 'G04BE03',
        '111111': 'H02AB02', '222222': 'J01CA04', '333333': 'K01AA02', '444444': 'L01XE01',
        '555555': 'M01AE01', '666666': 'N02BA01', '777777': 'O01AA01', '888888': 'P01AB01',
        '999999': 'Q01AA01', '121212': 'R03BA02', '131313': 'S01AA01', '141414': 'T01AA01',
        '151515': 'U01AA01', '161616': 'V01AA01'
    }

    state_list = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'WA', 'OR', 'CO', 'AZ', 'MA']

    # Create dictionaries with correct ICD entity types
    try:
        code_dict = CodeDict(
            atc_codes=atc_codes,
            icd_codes=icd_codes,
            rx_to_atc_map=rx_to_atc_map
        )

        age_dict = AgeDict(max_age=90, min_age=0, binsize=1)
        # Convert to string for consistency
        age_dict.vocab = [str(int(float(age))) if age not in DICT_DEFAULTS else age for age in age_dict.vocab]
        age_dict.labels_to_id = {(str(int(float(label))) if label not in DICT_DEFAULTS else label): idx
                                for label, idx in age_dict.labels_to_id.items()}
        age_dict.ids_to_label = {idx: (str(int(float(label))) if label not in DICT_DEFAULTS else label)
                                for idx, label in age_dict.ids_to_label.items()}

        sex_dict = SexDict(sex=['MALE', 'FEMALE'])
        state_dict = StateDict(states=state_list)

        print("✓ Created dictionaries with ICD entity types")
        print(f"  Entity types: {list(set(code_dict.entities))}")
        print(f"  Sample ICD codes: {list(code_dict.icd_codes)[:5]}")

    except Exception as e:
        print(f"✗ Dictionary creation error: {e}")
        return False

    # Create sample patients (from explain.ipynb)
    patients = [
        {
            'patient_id': 10000,
            'diagnoses': ['A00', 'B00', 'C00', 'D00', 'E00', 'F00', 'G00'],
            'diagnosis_dates': [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3), date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6), date(2021, 1, 7)],
            'drugs': ['111111', '222222', '333333', '444444', '555555', '666666', '777777'],
            'prescription_dates': [date(2021, 2, 1), date(2021, 2, 2), date(2021, 2, 3), date(2021, 2, 4), date(2021, 2, 5), date(2021, 2, 6), date(2021, 2, 7)],
            'birth_year': 1980,
            'sex': 'MALE',
            'patient_state': 'CA',
            'plos': 1
        },
        {
            'patient_id': 10001,
            'diagnoses': ['H00', 'J00', 'K00', 'L00', 'M00', 'N00', 'I10'],
            'diagnosis_dates': [date(2021, 1, 8), date(2021, 1, 9), date(2021, 1, 10), date(2021, 1, 11), date(2021, 1, 12), date(2021, 1, 13), date(2021, 1, 14)],
            'drugs': ['888888', '999999', '121212', '131313', '141414', '151515', '161616'],
            'prescription_dates': [date(2021, 2, 8), date(2021, 2, 9), date(2021, 2, 10), date(2021, 2, 11), date(2021, 2, 12), date(2021, 2, 13), date(2021, 2, 14)],
            'birth_year': 1975,
            'sex': 'FEMALE',
            'patient_state': 'NY',
            'plos': 0
        },
        {
            'patient_id': 10002,
            'diagnoses': ['E11.9', 'Z51.11', 'K21.0', 'M17.9', 'E78.5', 'N18.3', 'A00'],
            'diagnosis_dates': [date(2021, 3, 1), date(2021, 3, 2), date(2021, 3, 3), date(2021, 3, 4), date(2021, 3, 5), date(2021, 3, 6), date(2021, 3, 7)],
            'drugs': ['860975', '197361', '123456', '654321', '789012', '345678', '987654'],
            'prescription_dates': [date(2021, 4, 1), date(2021, 4, 2), date(2021, 4, 3), date(2021, 4, 4), date(2021, 4, 5), date(2021, 4, 6), date(2021, 4, 7)],
            'birth_year': 1990,
            'sex': 'MALE',
            'patient_state': 'TX',
            'plos': 1
        }
    ]

    # Add more patients to ensure sufficient data for train/val/test split
    additional_patients = []
    for i in range(3, 30):  # Create patients 10003 to 10029
        patient = {
            'patient_id': 10000 + i,
            'diagnoses': ['I10', 'E11.9', 'A00', 'B00'],  # Mix of ICD codes
            'diagnosis_dates': [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3), date(2021, 1, 4)],
            'drugs': ['111111', '222222', '333333', '444444'],
            'prescription_dates': [date(2021, 2, 1), date(2021, 2, 2), date(2021, 2, 3), date(2021, 2, 4)],
            'birth_year': 1980 + (i % 20),
            'sex': 'MALE' if i % 2 == 0 else 'FEMALE',
            'patient_state': state_list[i % len(state_list)],
            'plos': i % 2
        }
        additional_patients.append(patient)

    patients.extend(additional_patients)

    # Create Patient objects
    try:
        patient_objs = []
        for patient_data in patients:
            patient_obj = Patient(
                patient_id=patient_data['patient_id'],
                diagnoses=patient_data['diagnoses'],
                drugs=patient_data['drugs'],
                diagnosis_dates=patient_data['diagnosis_dates'],
                prescription_dates=patient_data['prescription_dates'],
                birth_year=patient_data['birth_year'],
                sex=patient_data['sex'],
                patient_state=patient_data['patient_state'],
                max_length=params['max_length'],
                code_embed=code_dict,
                sex_embed=sex_dict,
                age_embed=age_dict,
                state_embed=state_dict,
                mask_drugs=params['mask_drugs'],
                delete_temporary_variables=params['delete_temporary_variables'],
                split_sequence=params['split_sequence'],
                drop_duplicates=params['drop_duplicates'],
                converted_codes=params['converted_codes'],
                convert_rxcui_to_atc=params['convert_rxcui_to_atc'],
                keep_min_unmasked=params['min_unmasked'],
                max_masked_tokens=params['max_masked'],
                masked_lm_prob=params['masked_lm_prob'],
                truncate=params['truncate'],
                index_date=None,
                dynamic_masking=params['dynamic_masking'],
                min_observations=params['min_observations'],
                age_usage=params['age_usage'],
                use_cls=params['use_cls'],
                use_sep=params['use_sep'],
                valid_patient=params['valid_patient'],
                num_visits=None,
                combined_length=None,
                unpadded_length=None
            )
            patient_objs.append(patient_obj)

        print(f"✓ Created {len(patient_objs)} Patient objects")

        # Verify entity types in a sample patient
        sample_input = patient_objs[0].get_patient_data(
            evaluate=True,
            mask_dynamically=params['dynamic_masking'],
            code_embed=code_dict,
            min_unmasked=params['min_unmasked'],
            max_masked=params['max_masked'],
            masked_lm_prob=params['masked_lm_prob'],
            mask_drugs=params['mask_drugs']
        )

        print(f"✓ Sample patient input keys: {list(sample_input.keys())}")
        if 'entity_ids' in sample_input:
            entity_types = [code_dict.ids_to_entity.get(x.item(), 'UNK') for x in sample_input['entity_ids'][:10]]
            print(f"  Sample entity types: {entity_types}")

    except Exception as e:
        print(f"✗ Patient creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Split and create datasets
    try:
        indices = list(range(len(patient_objs)))
        train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

        train_patients = [patient_objs[i] for i in train_indices]
        val_patients = [patient_objs[i] for i in val_indices]
        test_patients = [patient_objs[i] for i in test_indices]

        # Create datasets
        train_dataset = PatientDataset(
            code_embed=code_dict,
            age_embed=age_dict,
            sex_embed=sex_dict,
            state_embed=state_dict,
            endpoint_dict=None,
            patients=train_patients,
            do_eval=True,
            max_length=params['max_length'],
            dynamic_masking=params['dynamic_masking'],
            min_unmasked=params['min_unmasked'],
            max_masked=params['max_masked'],
            masked_lm_prob=params['masked_lm_prob']
        )

        val_dataset = PatientDataset(
            code_embed=code_dict,
            age_embed=age_dict,
            sex_embed=sex_dict,
            state_embed=state_dict,
            endpoint_dict=None,
            patients=val_patients,
            do_eval=True,
            max_length=params['max_length'],
            dynamic_masking=params['dynamic_masking'],
            min_unmasked=params['min_unmasked'],
            max_masked=params['max_masked'],
            masked_lm_prob=params['masked_lm_prob']
        )

        test_dataset = PatientDataset(
            code_embed=code_dict,
            age_embed=age_dict,
            sex_embed=sex_dict,
            state_embed=state_dict,
            endpoint_dict=None,
            patients=test_patients,
            do_eval=True,
            max_length=params['max_length'],
            dynamic_masking=params['dynamic_masking'],
            min_unmasked=params['min_unmasked'],
            max_masked=params['max_masked'],
            masked_lm_prob=params['masked_lm_prob']
        )

        print(f"✓ Created datasets:")
        print(f"  Train: {len(train_dataset)} patients")
        print(f"  Validation: {len(val_dataset)} patients")
        print(f"  Test: {len(test_dataset)} patients")

    except Exception as e:
        print(f"✗ Dataset creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Save datasets
    try:
        os.makedirs('pretrain_stuff', exist_ok=True)

        train_output_path = 'pretrain_stuff/demo_train_patient_dataset.pt'
        val_output_path = 'pretrain_stuff/demo_val_patient_dataset.pt'
        test_output_path = 'pretrain_stuff/demo_test_patient_dataset.pt'

        train_dataset.save_dataset(path=train_output_path, with_patients=True, do_copy=True)
        val_dataset.save_dataset(path=val_output_path, with_patients=True, do_copy=True)
        test_dataset.save_dataset(path=test_output_path, with_patients=True, do_copy=True)

        print(f"✓ Saved datasets:")
        print(f"  Train: {train_output_path}")
        print(f"  Validation: {val_output_path}")
        print(f"  Test: {test_output_path}")

        # Verify the saved dataset has correct entity types
        print("\n=== Verification ===")
        loaded_train = PatientDataset.load_dataset(train_output_path)
        code_embed_loaded = loaded_train.code_embed
        print(f"Loaded entity types: {list(set(code_embed_loaded.entities))}")

        return True

    except Exception as e:
        print(f"✗ Dataset saving error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Regenerating training data with correct ICD entity types...")
    print("=" * 60)

    success = main()

    print("=" * 60)
    if success:
        print("🎉 Successfully regenerated training data with ICD entity types!")
        print("The datasets now contain proper 'icd' entity types for ClinVec integration.")
    else:
        print("❌ Failed to regenerate training data. Please check the errors above.")

    sys.exit(0 if success else 1)