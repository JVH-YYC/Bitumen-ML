#!/bin/bash
#SBATCH --mem=300000M
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00
#SBATCH --error=DOM9.err
#SBATCH --output=DOM9.out
python -u DOM9.py
