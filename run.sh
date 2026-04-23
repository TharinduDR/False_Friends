#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --mem=40G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home
huggingface-cli login --token

python token_classification.py train \
    --model_name xlm-roberta-base \
    --dataset_name false-friends/en_es_token \
    --output_dir ./ff_xlmr \
    --epochs 10 \
    --batch_size 16 \
    --lr 5e-5



