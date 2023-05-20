#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --partition=campus-gpu-bigmem
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH -e ./jobs/myjob.e%j
#SBATCH -o ./jobs/myjob.o%j 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ababjac@vols.utk.edu

cd $SLURM_SUBMIT_DIR
source $SCRATCHDIR/pyvenv/bin/activate
python drbert-copy.py

