#!/bin/sh
#SBATCH --job-name causalBERT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5g
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=genakim@umich.edu
#SBATCH -o ./slurm-outputs/%j.out
python CausalBert.py > slurm-outputs/output-$SLURM_JOBID.txt
