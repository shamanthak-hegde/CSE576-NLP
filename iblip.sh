#!/bin/bash
#SBATCH -p general
#SBATCH -t 2-00:0:00
#SBATCH --mem=50GB
#SBATCH -G a100:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shegde23@asu.edu
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE

module purge
module load mamba/latest
source activate vilt
cd /scratch/shegde23/NLP
python run_coco_instruct.py
