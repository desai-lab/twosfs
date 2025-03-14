#!/bin/bash
#
#SBATCH -p desai # Partition to submit to (comma separated)
#SBATCH -J slim_runs # Job name
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-0:10 # Runtime in D-HH:MM (or use minutes)
#SBATCH --mem 1000 # Memory in MB

python check_diversity.py
