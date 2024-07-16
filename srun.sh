#!/bin/bash

#SBATCH --job-name=array
#SBATCH --array=1-20
#SBATCH --time=0-4:00
#SBATCH -c 6
#SBATCH -n 1
#SBATCH -p htc
#SBATCH -q normal
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=ssganti1@asu.edu # Mail-to address
srun -c 6 ./the_work.sh $SLURM_ARRAY_TASK_ID
