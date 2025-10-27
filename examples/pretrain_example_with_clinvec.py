import subprocess
import sys
import os
import yaml

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_pretraining_with_clinvec():
    """
    Run the ExMed-BERT pretraining script with ClinVec integration.
    Parameters are loaded from config.yaml.
    """
    # Load configuration
    config = load_config()
    params = config['training_params_example']

    # Build the command with all parameters from config
    cmd = [
        sys.executable, "scripts/pretrain-exmed-bert-clinvec.py",
        params['training_data'],
        params['validation_data'],
        params['output_dir'],
        params['output_data_dir'],
        "--train-batch-size", str(params['train_batch_size']),
        "--eval-batch-size", str(params['eval_batch_size']),
        "--num-attention-heads", str(params['num_attention_heads']),
        "--num-hidden-layers", str(params['num_hidden_layers']),
        "--hidden-size", str(params['hidden_size']),
        "--intermediate-size", str(params['intermediate_size']),
        "--epochs", str(params['epochs']),
        "--max-steps", str(params['max_steps']),
        "--learning-rate", str(params['learning_rate']),
        "--gradient-accumulation-steps", str(params['gradient_accumulation_steps']),
        "--max-seq-length", str(params['max_seq_length']),
        "--seed", str(params['seed']),
        "--num-workers", str(params['num_workers']),
        "--logging-steps", str(params['logging_steps']),
        "--eval-steps", str(params['eval_steps']),
        "--save-steps", str(params['save_steps']),
        "--warmup-steps", str(params['warmup_steps']),
        "--initialization", params['initialization']
    ]

    # Add boolean flags
    if params['dynamic_masking']:
        cmd.append("--dynamic-masking")
    if params['plos']:
        cmd.append("--plos")

    # Add ClinVec integration parameters
    # Check if ClinVec parameters exist in config, otherwise use defaults
    clinvec_params = config.get('clinvec_params', {})

    # Add ClinVec directory if specified
    clinvec_dir = clinvec_params.get('clinvec_dir')
    if clinvec_dir:
        cmd.extend(["--clinvec-dir", str(clinvec_dir)])

    # Add vocabulary types as comma-separated string
    vocab_types = clinvec_params.get('vocab_types', ["icd9cm", "icd10cm", "phecode"])
    if vocab_types:
        vocab_string = ",".join(vocab_types)
        cmd.extend(["--vocab-types", vocab_string])

    # Add ClinVec flags
    if clinvec_params.get('use_clinvec', True):
        cmd.append("--use-clinvec")

    if clinvec_params.get('use_hierarchical_init', True):
        cmd.append("--use-hierarchical-init")

    if clinvec_params.get('resize_if_needed', True):
        cmd.append("--resize-if-needed")

    # Set PYTHONPATH to current directory so exmed_bert is found
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    print("Running pretraining script with ClinVec integration:")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    print(f"Pretraining script exited with code {result.returncode}")

if __name__ == "__main__":
    run_pretraining_with_clinvec()