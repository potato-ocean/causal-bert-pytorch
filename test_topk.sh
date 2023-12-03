#!/bin/sh
#SBATCH --job-name causalBERT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3g
#SBATCH --time=00:03:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=nrdavid@umich.edu
python ComputeTopKSimilarWords.py > slurm-outputs/output-$SLURM_JOBID.txt