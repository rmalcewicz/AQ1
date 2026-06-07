#!/bin/bash
#SBATCH --job-name=aq1_test
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --account=project_465003017
#SBATCH --output=/scratch/project_465003017/aq1_decoder/logs/test_%j.log

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.7

export PYTHONPATH=/scratch/project_465003017/aq1_decoder/packages:$PYTHONPATH

cd /scratch/project_465003017/aq1_decoder
python3 train.py --data data/d3_zbasis_test.h5 --lr 3e-4 --epochs 5 --run_name test_lr3e4
