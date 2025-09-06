import sys
sys.path.insert(0, '/home/sersasj/RSNA-IAD-Codebase/nnUNet')

from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess

# Set dataset ID
dataset_id = 901

# Extract fingerprints (with integrity check)
extract_fingerprints([dataset_id], check_dataset_integrity=True, verbose=True)

# Plan experiments
plans_identifier = plan_experiments([dataset_id], gpu_memory_target_in_gb=8)  # Adjust GPU memory as needed

# Preprocess
preprocess([dataset_id], plans_identifier=plans_identifier, configurations=['3d_fullres'], num_processes=[8])