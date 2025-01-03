#!/bin/bash

#SBATCH --job-name=exp1
#SBATCH --account=scw2258

# job stdout file. The '%J' to Slurm is replaced with the job number.
#SBATCH --output=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/stdout/stdout_%J.log
#SBATCH --error=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/stderr/stderr_%J.log

# Number of GPUs to allocate (don't forget to select a partition with GPUs)
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:1
### SBATCH -t 0-00:00

# Number of CPU cores per task to allocate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

cuda=CUDA/11.7
conda=anaconda/2024.02
env=arrg_img2text

module load $cuda
echo "Loaded $cuda"

module load $conda
source activate
conda activate $env
echo "Loaded $conda, env: $env"
nvcc -V

python /scratch/c.c21051562/workspace/arrg_img2text/0_img_cls_effusion.py --from_bash --config_file exp1_imgcls_arcca.yaml

python /scratch/c.c21051562/workspace/arrg_img2text/test_email.py

# sbatch /scratch/c.c21051562/workspace/arrg_img2text/run_arcca.sh
# scontrol show job JOBID
# scontrol show job JOBID | grep NodeList
# scancel JOBID

# Server side:
# tensorboard --logdir=/home/yuxiang/liao/workspace/arrg_img2text/outputs/logs --port=6006

# Check process and kill
# ps aux | grep <进程名>
# kill <PID>

# Client side:
# ssh -L 6007:localhost:6006 yuxiang@10.97.37.97
# tensorboard --logdir=/home/yuxiang/liao/workspace/arrg_img2text/outputs/logs --port=6006

# ssh -L 6007:localhost:6006 -J c.c21051562@hawklogin.cf.ac.uk c.c21051562@ccs2111
# tensorboard --logdir=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs