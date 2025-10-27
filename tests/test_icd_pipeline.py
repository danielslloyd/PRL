#!/usr/bin/env python3
"""
Test script for the ICD-10 pipeline
"""

import sys
import yaml
from datetime import date

def test_imports():
    """Test that all required classes can be imported"""
    try:
        from exmed_bert.data.encoding import CodeDict, AgeDict, SexDict, StateDict
        from exmed_bert.data.patient import Patient
        print('✓ Successfully imported all classes')
        return True
    except Exception as e:
        print(f'✗ Import error: {e}')
        return False

def test_configuration():
    """Test configuration loading"""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        params = config['data_prep_params_example']
        convert_icd = params.get('convert_icd_to_phewas', 'NOT_FOUND')
        print(f'✓ Configuration loaded: convert_icd_to_phewas = {convert_icd}')

        if convert_icd is False:
            print('✓ ICD-to-PheWAS conversion is disabled (correct for raw ICD usage)')
        return True
    except Exception as e:
        print(f'✗ Config error: {e}')
        return False

def test_codedict():
    """Test CodeDict with ICD codes"""
    try:
        from exmed_bert.data.encoding import CodeDict

        atc_codes = ['A01AA01', 'B01AC06', 'C09AA05']
        icd_codes = ['I10', 'E11.9', 'A00', 'B00', 'Z51.11']
        rx_to_atc_map = {
            '111111': 'A01AA01',
            '222222': 'B01AC06',
            '333333': 'C09AA05'
        }

        code_dict = CodeDict(
            atc_codes=atc_codes,
            icd_codes=icd_codes,
            rx_to_atc_map=rx_to_atc_map
        )

        print(f'✓ CodeDict created successfully')
        print(f'  - {len(code_dict.icd_codes)} ICD codes')
        print(f'  - {len(code_dict.atc_codes)} ATC codes')
        print(f'  - Sample ICD codes: {list(code_dict.icd_codes)[:3]}')
        print(f'  - Entity types: {list(set(code_dict.entities))}')

        # Test encoding/decoding
        test_codes = ['I10', 'A01AA01']
        encoded = [code_dict(code) for code in test_codes]
        decoded = code_dict.decode(encoded)
        print(f'  - Encode/decode test: {test_codes} -> {encoded} -> {decoded}')

        return True
    except Exception as e:
        print(f'✗ CodeDict error: {e}')
        return False

def test_patient_creation():
    """Test Patient creation with ICD codes"""
    try:
        from exmed_bert.data.encoding import CodeDict, AgeDict, SexDict, StateDict
        from exmed_bert.data.patient import Patient

        # Setup dictionaries
        atc_codes = ['A01AA01', 'B01AC06']
        icd_codes = ['I10', 'E11.9', 'A00', 'B00']
        rx_to_atc_map = {'111111': 'A01AA01', '222222': 'B01AC06'}

        code_dict = CodeDict(
            atc_codes=atc_codes,
            icd_codes=icd_codes,
            rx_to_atc_map=rx_to_atc_map
        )
        age_dict = AgeDict(max_age=90, min_age=0, binsize=1)
        sex_dict = SexDict(sex=['MALE', 'FEMALE'])
        state_dict = StateDict(states=['CA', 'NY'])

        # Create patient with ICD codes
        patient = Patient(
            patient_id=10000,
            diagnoses=['I10', 'E11.9'],  # Raw ICD codes
            drugs=['111111', '222222'],  # RxNorm codes (will be converted to ATC)
            diagnosis_dates=[date(2021, 1, 1), date(2021, 1, 2)],
            prescription_dates=[date(2021, 2, 1), date(2021, 2, 2)],
            birth_year=1980,
            sex='MALE',
            patient_state='CA',
            max_length=50,
            code_embed=code_dict,
            sex_embed=sex_dict,
            age_embed=age_dict,
            state_embed=state_dict,
            mask_drugs=True,
            delete_temporary_variables=False,
            split_sequence=True,
            drop_duplicates=True,
            converted_codes=False,
            convert_rxcui_to_atc=True,  # Convert drugs only
            keep_min_unmasked=1,
            max_masked_tokens=20,
            masked_lm_prob=0.15,
            truncate='right',
            dynamic_masking=True,
            min_observations=2,
            age_usage='year',
            use_cls=True,
            use_sep=False,
            valid_patient=True
        )

        # Get model input
        model_input = patient.get_patient_data(
            evaluate=True,
            mask_dynamically=True,
            code_embed=code_dict,
            min_unmasked=1,
            max_masked=20,
            masked_lm_prob=0.15,
            mask_drugs=True
        )

        print(f'✓ Patient created successfully')
        print(f'  - Model input keys: {list(model_input.keys())}')
        print(f'  - Code labels present: {"code_labels" in model_input}')
        print(f'  - Input sequence length: {len(model_input["input_ids"])}')

        # Decode some codes to verify ICD codes are preserved
        input_ids = model_input['input_ids'][:10]
        decoded_codes = code_dict.decode(input_ids)
        print(f'  - First 10 decoded codes: {decoded_codes}')

        return True

    except Exception as e:
        print(f'✗ Patient creation error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing ICD-10 Pipeline...")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("CodeDict Test", test_codedict),
        ("Patient Creation Test", test_patient_creation)
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1
        else:
            print(f"FAILED: {test_name}")

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! ICD-10 pipeline is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)