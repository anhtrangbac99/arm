#!/bin/bash 
#SBATCH -o outputs/train_image_gopro/job_%j.output
#SBATCH -e errors/train_image_gopro/job_%j.error
#SBATCH -p PA100q
#SBATCH --gres=gpu:2
#SBATCH -n 2
#SBATCH -c 2

module load cuda11.1/toolkit 
module load cuda11.1/blas/11.1.1 
source activate science
python train_transformer.py 