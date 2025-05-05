#!/bin/bash

#SBATCH --job-name=5_preprocess_interpretcxr_full_text_img518
#SBATCH --account=scw2258

# Job stdout file. The '%J' = job number. %x = job name
#SBATCH --output=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/%x/stdout/stdout_%J.log
#SBATCH --error=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/%x/stderr/stderr_%J.log

#SBATCH --partition=compute
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
python /scratch/c.c21051562/workspace/arrg_img2text/5_fsdp_peft_full_text.py \
    --from_bash \
    --config_file /scratch/c.c21051562/workspace/arrg_img2text/config/sunbird/5_fsdp_peft_full_text.yaml \
    --output_name $SLURM_JOB_NAME \
    --jobid $SLURM_JOB_ID \
    --run_mode preprocess \

echo "Script finished."


python /scratch/c.c21051562/workspace/test_email.py --from_bash --subject "Sunbird Done: $SLURM_JOB_NAME"

# sbatch /scratch/c.c21051562/workspace/arrg_img2text/scripts/preprocess_sunbird.sh
# scontrol show job JOBID
# scontrol show job JOBID | grep NodeList
# scancel JOBID

