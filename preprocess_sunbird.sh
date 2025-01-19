#!/bin/bash

#SBATCH --job-name=1_fast_000test_preprocess_data
#SBATCH --account=scw2258

# job stdout file. The '%J' to Slurm is replaced with the job number. %x = Job name
#SBATCH --output=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/%x/stdout/stdout_%J.log
#SBATCH --error=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/%x/stderr/stderr_%J.log

# Number of GPUs to allocate (don't forget to select a partition with GPUs)
#SBATCH --partition=compute
### SBATCH --gres=gpu:2
### SBATCH -t 0-00:00

# Number of CPU cores per task to allocate, (maximun of 8 cpus for 2 gpus)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

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

echo "Running script ... (job: $SLURM_JOB_NAME $SLURM_JOB_ID)"
python /scratch/c.c21051562/workspace/arrg_img2text/1_cls_effu_notallimg_fast.py \
    --from_bash \
    --config_file /scratch/c.c21051562/workspace/arrg_img2text/config/sunbird/1_imgcls_notallimg_fast.yaml \
    --output_name $SLURM_JOB_NAME \
    --jobid $SLURM_JOB_ID \
    --preprocess_dataset
echo "Script finished."


python /scratch/c.c21051562/workspace/test_email.py --from_bash --subject "Sunbird Done: $SLURM_JOB_NAME"

# sbatch /scratch/c.c21051562/workspace/arrg_img2text/preprocess_sunbird.sh
# scontrol show job JOBID
# scontrol show job JOBID | grep NodeList
# scancel JOBID

