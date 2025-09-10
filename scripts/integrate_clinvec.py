#!/usr/bin/env python3
"""
Example script: Integrate ClinVec embeddings with ExMed-BERT

Usage:
    python scripts/integrate_clinvec.py --clinvec_dir path/to/ClinVec --config_file config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from exmed_bert.models.config import ExMedBertConfig
from exmed_bert.models.model import ExMedBertModel
from exmed_bert.data.encoding import CodeDict
from exmed_bert.utils.clinvec_integration import (
    integrate_clinvec_with_exmedbert,
    update_exmedbert_init_weights,
    load_for_exmedbert_pretraining,
    load_for_exmedbert_finetuning
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_code_dict():
    """Create example code dictionary for testing"""
    # This would normally come from your data preprocessing
    example_codes = [
        "PAD", "UNK", "CLS", "SEP", "MASK",  # Special tokens
        "250.0", "250.00", "E11", "E119",    # Diabetes codes
        "401.9", "I10", "I15",               # Hypertension codes
        "272.0", "E78", "E785",              # Dyslipidemia codes
        "V27.0", "Z39", "Z390",              # Normal delivery codes
    ]
    
    code_dict = CodeDict()
    for code in example_codes:
        code_dict.add_item(code)
    
    logger.info(f"Created example CodeDict with {len(code_dict)} codes")
    return code_dict


def test_integration(clinvec_dir: str, vocab_types: list = None):
    """Test ClinVec integration with a small ExMed-BERT model"""
    
    if vocab_types is None:
        vocab_types = ["icd9cm", "icd10cm"]
    
    logger.info("=== Testing ClinVec Integration ===")
    
    # Create small test configuration
    config = ExMedBertConfig(
        code_vocab_size=100,
        hidden_size=128,  # Small for testing
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32
    )
    
    # Create model and code dictionary
    model = ExMedBertModel(config)
    code_dict = create_example_code_dict()
    
    logger.info(f"Model embedding dimension: {config.hidden_size}")
    logger.info(f"Model vocabulary size: {config.code_vocab_size}")
    
    # Integrate ClinVec embeddings
    stats = integrate_clinvec_with_exmedbert(
        model=model,
        code_dict=code_dict,
        clinvec_dir=clinvec_dir,
        vocab_types=vocab_types,
        resize_if_needed=True,
        verbose=True
    )
    
    # Update initialization method
    update_exmedbert_init_weights(model)
    
    # Test that embeddings are preserved after init_weights call
    logger.info("\n=== Testing Preservation After init_weights ===")
    original_weight = model.embeddings.code_embeddings.weight.clone()
    model.init_weights()  # Should preserve ClinVec embeddings
    
    weight_changed = not torch.equal(original_weight, model.embeddings.code_embeddings.weight)
    if weight_changed:
        logger.warning("Embeddings were modified after init_weights - preservation failed!")
    else:
        logger.info("✓ Embeddings preserved after init_weights call")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Integrate ClinVec embeddings with ExMed-BERT")
    parser.add_argument("--clinvec_dir", required=True, help="Path to ClinVec dataset directory")
    parser.add_argument("--vocab_types", nargs="+", default=["icd9cm", "icd10cm"], 
                       help="Vocabulary types to load")
    parser.add_argument("--test_only", action="store_true", 
                       help="Run test integration only")
    
    args = parser.parse_args()
    
    # Verify ClinVec directory exists
    clinvec_path = Path(args.clinvec_dir)
    if not clinvec_path.exists():
        logger.error(f"ClinVec directory not found: {clinvec_path}")
        sys.exit(1)
    
    required_files = ["ClinGraph_nodes.csv", "ClinVec_icd9cm.csv", "ClinVec_icd10cm.csv"]
    for file in required_files:
        if not (clinvec_path / file).exists():
            logger.error(f"Required file not found: {clinvec_path / file}")
            sys.exit(1)
    
    if args.test_only:
        # Run test integration
        stats = test_integration(args.clinvec_dir, args.vocab_types)
        
        print("\n=== Integration Results ===")
        for vocab, count in stats.items():
            print(f"{vocab}: {count} embeddings loaded")
        
        total = sum(stats.values())
        if total > 0:
            print(f"✓ Successfully integrated {total} ClinVec embeddings!")
        else:
            print("⚠ No embeddings were loaded - check code dictionary compatibility")
    
    else:
        print("For production use, integrate this into your training script:")
        print("from exmed_bert.utils.clinvec_integration import load_for_exmedbert_pretraining")
        print("stats = load_for_exmedbert_pretraining(model, code_dict, 'path/to/ClinVec')")


if __name__ == "__main__":
    main()