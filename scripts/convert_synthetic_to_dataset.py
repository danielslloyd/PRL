#!/usr/bin/env python
"""
Convert synthetic patient JSON data to PatientDataset format.

This script takes the JSON output from generate_synthetic_patients.py and
converts it into the PatientDataset format expected by the ExMed-BERT
pretraining scripts.

Usage:
    python scripts/convert_synthetic_to_dataset.py \
        --input data/synthetic_patients.json \
        --output pretrain_stuff/synthetic_train.pt \
        --split train
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import typer
from typer import Option

from exmed_bert.data.dataset import PatientDataset
from exmed_bert.data.encoding import AgeDict, CodeDict, EndpointDict, SexDict, StateDict
from exmed_bert.data.patient import Patient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_vocabulary_dicts(patients_data: List[dict]) -> tuple:
    """
    Create vocabulary dictionaries from patient data.

    Args:
        patients_data: List of patient dictionaries from JSON

    Returns:
        Tuple of (code_dict, age_dict, sex_dict, state_dict, endpoint_dict)
    """
    logger.info("Creating vocabulary dictionaries from patient data...")

    # Collect all unique codes
    all_icd_codes = set()
    all_atc_codes = set()
    all_states = set()

    for patient in patients_data:
        all_icd_codes.update(patient['diagnoses'])
        all_atc_codes.update(patient['drugs'])
        all_states.add(patient['patient_state'])

    logger.info(f"  Found {len(all_icd_codes)} unique ICD-10 codes")
    logger.info(f"  Found {len(all_atc_codes)} unique ATC5 codes")
    logger.info(f"  Found {len(all_states)} unique states")

    # Create CodeDict
    code_dict = CodeDict(
        atc_codes=sorted(list(all_atc_codes)),
        icd_codes=sorted(list(all_icd_codes)),
        rx_to_atc_map=None  # Already using ATC codes
    )

    # Create other embedding dicts
    age_dict = AgeDict(max_age=110, min_age=0, binsize=1.0)  # Age in years
    sex_dict = SexDict()
    state_dict = StateDict(states=sorted(list(all_states)))
    endpoint_dict = EndpointDict([])  # No endpoints for pretraining

    logger.info(f"  Code vocabulary size: {len(code_dict)}")
    logger.info(f"  Age vocabulary size: {len(age_dict)}")
    logger.info(f"  Sex vocabulary size: {len(sex_dict)}")
    logger.info(f"  State vocabulary size: {len(state_dict)}")

    return code_dict, age_dict, sex_dict, state_dict, endpoint_dict


def convert_json_to_patients(
    patients_data: List[dict],
    code_dict: CodeDict,
    age_dict: AgeDict,
    sex_dict: SexDict,
    state_dict: StateDict,
    max_length: int = 512,
    dynamic_masking: bool = True,
    min_observations: int = 5,
) -> List[Patient]:
    """
    Convert JSON patient data to Patient objects.

    Args:
        patients_data: List of patient dictionaries from JSON
        code_dict: CodeDict instance
        age_dict: AgeDict instance
        sex_dict: SexDict instance
        state_dict: StateDict instance
        max_length: Maximum sequence length
        dynamic_masking: Whether to use dynamic masking
        min_observations: Minimum observations required per patient

    Returns:
        List of Patient objects
    """
    logger.info(f"Converting {len(patients_data)} patients from JSON to Patient objects...")

    patients = []
    failed_count = 0

    for i, patient_data in enumerate(patients_data):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(patients_data)} patients...")

        try:
            # Convert date strings to date objects
            diagnosis_dates = [
                datetime.strptime(d, "%Y%m%d").date()
                for d in patient_data['diagnosis_dates']
            ]
            prescription_dates = [
                datetime.strptime(d, "%Y%m%d").date()
                for d in patient_data['prescription_dates']
            ]

            # Create Patient object
            patient = Patient(
                patient_id=patient_data['patient_id'],
                diagnoses=patient_data['diagnoses'],
                drugs=patient_data['drugs'],
                diagnosis_dates=diagnosis_dates,
                prescription_dates=prescription_dates,
                birth_year=patient_data['birth_year'],
                sex=patient_data['sex'],
                patient_state=patient_data['patient_state'],
                # Configuration parameters
                max_length=max_length,
                code_embed=code_dict,
                sex_embed=sex_dict,
                age_embed=age_dict,
                state_embed=state_dict,
                mask_drugs=True,
                delete_temporary_variables=True,
                split_sequence=True,
                drop_duplicates=True,
                converted_codes=True,  # Codes are already in final format
                convert_rxcui_to_atc=False,  # Already ATC codes
                keep_min_unmasked=1,
                max_masked_tokens=20,
                masked_lm_prob=0.15,
                truncate='right',
                dynamic_masking=dynamic_masking,
                min_observations=min_observations,
                age_usage='year',
                use_cls=False,
                use_sep=False,
                endpoint_labels=None,  # No endpoints for pretraining
            )

            patients.append(patient)

        except Exception as e:
            failed_count += 1
            logger.debug(f"  Failed to process patient {patient_data['patient_id']}: {e}")
            continue

    logger.info(f"Successfully converted {len(patients)} patients")
    if failed_count > 0:
        logger.warning(f"Failed to convert {failed_count} patients (likely too few observations)")

    return patients


def create_train_val_split(
    patients: List[Patient],
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple:
    """
    Split patients into training and validation sets.

    Args:
        patients: List of Patient objects
        train_ratio: Ratio of training data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_patients, val_patients)
    """
    import random
    random.seed(seed)

    # Shuffle patients
    shuffled = patients.copy()
    random.shuffle(shuffled)

    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_patients = shuffled[:split_idx]
    val_patients = shuffled[split_idx:]

    logger.info(f"Split into {len(train_patients)} training and {len(val_patients)} validation patients")

    return train_patients, val_patients


def main(
    input_json: Path = Option(..., "--input", "-i", help="Input JSON file with synthetic patients"),
    output_path: Path = Option(..., "--output", "-o", help="Output .pt file path"),
    split: str = Option("all", "--split", "-s", help="Dataset split: 'train', 'val', or 'all'"),
    max_length: int = Option(512, "--max-length", help="Maximum sequence length"),
    train_ratio: float = Option(0.8, "--train-ratio", help="Training set ratio (if split=all)"),
    dynamic_masking: bool = Option(True, "--dynamic-masking/--no-dynamic-masking", help="Use dynamic masking"),
    seed: int = Option(42, "--seed", help="Random seed"),
):
    """
    Convert synthetic patient JSON to PatientDataset format.
    """
    logger.info("=" * 60)
    logger.info("CONVERTING SYNTHETIC PATIENTS TO PATIENTDATASET")
    logger.info("=" * 60)

    # Load JSON data
    logger.info(f"Loading data from {input_json}...")
    with open(input_json, 'r') as f:
        patients_data = json.load(f)
    logger.info(f"Loaded {len(patients_data)} patients from JSON")

    # Create vocabulary dictionaries
    code_dict, age_dict, sex_dict, state_dict, endpoint_dict = create_vocabulary_dicts(patients_data)

    # Convert to Patient objects
    patients = convert_json_to_patients(
        patients_data=patients_data,
        code_dict=code_dict,
        age_dict=age_dict,
        sex_dict=sex_dict,
        state_dict=state_dict,
        max_length=max_length,
        dynamic_masking=dynamic_masking,
    )

    # Handle splitting
    if split == "all":
        # Create both train and validation datasets
        train_patients, val_patients = create_train_val_split(
            patients, train_ratio=train_ratio, seed=seed
        )

        # Determine output paths
        output_dir = output_path.parent
        output_stem = output_path.stem
        train_path = output_dir / f"{output_stem}_train.pt"
        val_path = output_dir / f"{output_stem}_val.pt"

        # Create datasets
        logger.info("Creating training dataset...")
        train_dataset = PatientDataset(
            code_embed=code_dict,
            age_embed=age_dict,
            sex_embed=sex_dict,
            state_embed=state_dict,
            endpoint_dict=endpoint_dict,
            patients=train_patients,
            max_length=max_length,
            do_eval=True,
            mask_substances=True,
            dynamic_masking=dynamic_masking,
        )

        logger.info("Creating validation dataset...")
        val_dataset = PatientDataset(
            code_embed=code_dict,
            age_embed=age_dict,
            sex_embed=sex_dict,
            state_embed=state_dict,
            endpoint_dict=endpoint_dict,
            patients=val_patients,
            max_length=max_length,
            do_eval=True,
            mask_substances=True,
            dynamic_masking=dynamic_masking,
        )

        # Save datasets
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving training dataset to {train_path}...")
        torch.save(train_dataset, train_path)
        logger.info(f"Saving validation dataset to {val_path}...")
        torch.save(val_dataset, val_path)

        logger.info("=" * 60)
        logger.info("CONVERSION COMPLETE!")
        logger.info(f"  Training dataset: {train_path}")
        logger.info(f"    - Patients: {len(train_dataset)}")
        logger.info(f"  Validation dataset: {val_path}")
        logger.info(f"    - Patients: {len(val_dataset)}")
        logger.info("=" * 60)

    else:
        # Create single dataset
        logger.info(f"Creating {split} dataset...")
        dataset = PatientDataset(
            code_embed=code_dict,
            age_embed=age_dict,
            sex_embed=sex_dict,
            state_embed=state_dict,
            endpoint_dict=endpoint_dict,
            patients=patients,
            max_length=max_length,
            do_eval=True,
            mask_substances=True,
            dynamic_masking=dynamic_masking,
        )

        # Save dataset
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset to {output_path}...")
        torch.save(dataset, output_path)

        logger.info("=" * 60)
        logger.info("CONVERSION COMPLETE!")
        logger.info(f"  Dataset: {output_path}")
        logger.info(f"  Patients: {len(dataset)}")
        logger.info(f"  Vocabulary size: {len(code_dict)}")
        logger.info("=" * 60)


if __name__ == "__main__":
    typer.run(main)
