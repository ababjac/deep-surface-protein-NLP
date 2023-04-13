#!/bin/bash
#SBATCH -A ISAAC-UTK0196
#SBATCH --partition=condo-semrich-temp
#SBATCH --qos=condo
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=5-00:00:00
#SBATCH -e ./jobs/myjob.e%j
#SBATCH -o ./jobs/myjob.o%j 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ababjac@vols.utk.edu

cd $SLURM_SUBMIT_DIR
source $SCRATCHDIR/pyvenv/bin/activate
python aminoBERT_combo.py

