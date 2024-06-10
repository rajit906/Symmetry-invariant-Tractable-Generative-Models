#!/bin/bash
# example job script for anaconda
#$ -N test_conda -cwd
#$ -l h_rt=00:01:00 # only runs for 1  mins!
#$ -l h_vmem=8G
#$ -m bea -M s2592586@ed.ac.uk # CHANGE THIS TO YOUR EMAIL ADDRESS

. /etc/profile.d/modules.sh
module load anaconda/2024.02 # this loads a specific version of anaconda
conda activate symmetry # this starts the 'symmetry' environment

python experiment_1.py 

