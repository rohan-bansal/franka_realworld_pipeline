#!/bin/bash
#SBATCH --job-name=mimicgen_download_all
#SBATCH --output=/coc/flash7/nkra3/logs/sbatch_out/phd_project/processing/mimicgen_download_all.out
#SBATCH --error=/coc/flash7/nkra3/logs/sbatch_err/phd_project/processing/mimicgen_download_all.err
#SBATCH --partition=overcap
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node="a40:1"
#SBATCH --exclude="clippy"
#SBATCH --exclude="chappie"
#SBATCH --mem-per-gpu=80G

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
source /nethome/nkra3/flash7/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate robomimic-dev


cd /nethome/nkra3/flash7/phd_project/mimicgen/mimicgen


srun -u python -u scripts/download_datasets.py --dataset_type core --tasks all 