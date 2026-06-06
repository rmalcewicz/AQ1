#!/bin/bash

#SBATCH --job-name=train_aq1

#SBATCH --partition=standard-g

#SBATCH --nodes=1

#SBATCH --gpus-per-node=8

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16

#SBATCH --mem=120G

#SBATCH --time=24:00:00

#SBATCH --account=project_465003017

#SBATCH --output=/scratch/project_465003017/aq1_decoder/logs/train_%j.log



module purge

module use /appl/local/csc/modulefiles

module load pytorch/2.7



export PYTHONPATH=/scratch/project_465003017/aq1_decoder/packages:$PYTHONPATH



cd /scratch/project_465003017/aq1_decoder

python3 train.py

