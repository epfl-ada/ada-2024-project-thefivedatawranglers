#!/bin/bash

### Job name
#SBATCH --job-name=ml_train

### Time your job needs to execute, e. g. 30 min
#SBATCH --time=12:00:00

### Memory your job needs per node, e. g. 500 MB
#SBATCH --mem=40GB

### Number of threads to use, e. g. 24
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1

#SBATCH --output=/home/querfurth/ada/slurm/training_NN.%J.txt

### Change to working directory
cd /home/querfurth/ada/ada-2024-project-thefivedatawranglers
source /home/querfurth/venvs/ml2/bin/activate


# MODEL_NAME = "FPN"
# BACKBONE = "resnet18"
# ORIGINAL_ONLY = False  # Use additional datasets
# USE_AUGMENTATIONS = False  # Use full augmentations, otherwise only normalization
# MODEL_WEIGHTS = None  # Use pre-trained weights that were trained on the large satellite image dataset collection
# VERBOSE = False

python train_rating_prediction.py
# SVM Final eval: ML_SVM_FINAL.52523969.tx