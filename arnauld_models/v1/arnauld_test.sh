#!/bin/bash

#SBATCH --job-name=training
#SBATCH --output=/groups/CS156b/2024/clownmonkeys/arnauld_models/logs/%x_%j_std.out
#SBATCH --error=/groups/CS156b/2024/clownmonkeys/arnauld_models/logs/%x_%j_std.err
#SBATCH -A CS156b
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8G
#SBATCH --gres gpu:1
# SBATCH --mail-user=amartin7@caltech.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

module load cuda
module load gcc

source /groups/CS156b/2024/clownmonkeys/packages/anaconda3/bin/activate
conda deactivate
conda activate base

# set env vars
export WANDB_API_KEY="834807c3ce19310795bf38319e568644792946b0"

cd /central/groups/CS156b/2024/clownmonkeys

python arnauld_models/support_devices.py