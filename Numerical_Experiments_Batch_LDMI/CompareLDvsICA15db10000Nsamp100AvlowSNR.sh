#!/bin/bash
#SBATCH --job-name=ldvsica # Job name
#SBATCH --nodes=1 # Run on a single node
#SBATCH --ntasks-per-node=1 # Run a single task on each node
#SBATCH --partition=ai # Run in ai queue
#SBATCH --qos=ai # Run in qos (ai)
#SBATCH --account=ai # Run account (ai)
#SBATCH --time=72:0:0 # Time limit days-hours:minutes:seconds
#SBATCH --output=ldvsica-%j.out # Standard output and error log
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alperdogan@ku.edu.tr # Where to send mail
##SBATCH --gres=gpu:1
##SBATCH --constraint="tesla_t4"
##SBATCH --constraint="tesla_v100"
#SBATCH --mem=128G


module load anaconda/3.6

module load cuda/11.0
module load cudnn/8.0.4/cuda-11.0
module load python/3.7.4
module load gcc

source activate atepython

python -m pip install --no-cache --user mne 

# PRETRAINING

python CompareLDandICAscriptlowSNR.py --N 10000 --NumberofIterations 25000 --NumAverages 100 --SNR 15 --epsv 0.001 --outname ldvsicaresults/SIR15Av100N10000It20000New2.pickle


#sleep 100