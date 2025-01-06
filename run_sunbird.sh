#!/bin/bash

#SBATCH --job-name=0_img_cls_effusion_notallimg
#SBATCH --account=scw2258

# job stdout file. The '%J' to Slurm is replaced with the job number.
#SBATCH --output=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/stdout/stdout_%J.log
#SBATCH --error=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/stderr/stderr_%J.log

# Number of GPUs to allocate (don't forget to select a partition with GPUs)
#SBATCH --partition=accel_ai
#SBATCH --gres=gpu:1
### SBATCH -t 0-00:00

# Number of CPU cores per task to allocate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

cuda=CUDA/12.4
conda=anaconda/2024.06
env=arrg_img2text

module load $cuda
echo "Loaded $cuda"

module load $conda
source activate
conda activate $env
echo "Loaded $conda, env: $env"
nvcc -V

cfg_dir=/scratch/c.c21051562/workspace/arrg_img2text/config/sunbird

# python /scratch/c.c21051562/workspace/arrg_img2text/0_img_cls_effusion_notallimg.py --from_bash --config_file $cfg_dir/0_imgcls_notallimg.yaml
python /scratch/c.c21051562/workspace/arrg_img2text/1_img_cls_effusion_notallimg_attpool.py --from_bash --config_file $cfg_dir/1_imgcls_notallimg_attpool.yaml

python /scratch/c.c21051562/workspace/test_email.py --from_bash --subject "Done: 1_imgcls_notallimg_attpool"

# sbatch /scratch/c.c21051562/workspace/arrg_img2text/run_sunbird.sh
# scontrol show job JOBID
# scontrol show job JOBID | grep NodeList
# scancel JOBID

# tensorboard --logdir=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs --port=6006

# Check process and kill
# ps aux | grep <进程名>
# kill <PID>


# ssh -L 6007:localhost:6006 c.c21051562@sunbird.swansea.ac.uk
# tensorboard --logdir=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs --port=6006