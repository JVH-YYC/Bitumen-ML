#!/bin/bash
#SBATCH --mem=300000M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00
#SBATCH --error=DOM2.err
#SBATCH --output=DOM2.out
python -u DOM2.py