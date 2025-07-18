


import subprocess
import sys
import os

def run_pretraining():
    """
    Run the ExMed-BERT pretraining script from within pretrain.py.
    Adjust the paths and parameters as needed for your setup.
    """
    # Set your paths here
    training_data = "pretrain_stuff/demo_train_patient_dataset.pt"
    validation_data = "pretrain_stuff/demo_val_patient_dataset.pt"
    output_dir = "output/pretrain"
    output_data_dir = "output/pretrain_data"
    epochs = "5"  # Change as needed

    # Build the command
    cmd = [
        sys.executable, "scripts/pretrain-exmed-bert.py",
        training_data,
        validation_data,
        output_dir,
        output_data_dir,
        "--epochs", epochs
    ]

    # Set PYTHONPATH to current directory so exmed_bert is found
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    print("Running pretraining script with command:\n", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    print(f"Pretraining script exited with code {result.returncode}")

if __name__ == "__main__":
    run_pretraining()