#!/bin/bash

#SBATCH --job-name=gen_tier2

#SBATCH --partition=small

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=16

#SBATCH --mem=32G

#SBATCH --time=04:00:00

#SBATCH --account=project_465003017

#SBATCH --output=/scratch/project_465003017/aq1_decoder/logs/gen_tier2_%j.log



module purge

module use /appl/local/csc/modulefiles

module load pytorch/2.7

export PYTHONPATH=/scratch/project_465003017/aq1_decoder/packages:$PYTHONPATH

cd /scratch/project_465003017/aq1_decoder

python3 generate_data_iqm_full.py

