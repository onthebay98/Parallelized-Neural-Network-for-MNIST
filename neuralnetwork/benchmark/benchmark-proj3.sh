#!/bin/bash
#
#SBATCH --mail-user=bay@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=proj3_speedup
#SBATCH --output=./slurm/out/%j.%N.stdout
#SBATCH --error=./slurm/out/%j.%N.stderr
#SBATCH --chdir=/home/bay/project-3-onthebay98/proj3/benchmark
#SBATCH --partition=debug 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=900
#SBATCH --exclusive
#SBATCH --time=800:00

python3 speedup.py
