#!/bin/bash
#SBATCH -J create_abinitio_targets
#SBATCH -p standard
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --account dmobley_lab
#SBATCH --export ALL
#SBATCH --constraint=fastscratch
#SBATCH -o create_abinitio_targets.out
#SBATCH -e create_abinitio_targets.err

date
hostname

source ~/.bashrc
conda activate fb_196_ic_0318    

python build_abinitio_targets_group.py ../../sage-2.2.0/02_curate-data/output/optimization-training-set.json fb-fit/
