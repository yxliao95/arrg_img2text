#!/bin/bash

#SBATCH --job-name=1_imgcls_notallimg_fast_try3
#SBATCH --account=scw2258

# job stdout file. The '%J' to Slurm is replaced with the job number. %x = Job name
#SBATCH --output=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/%x/stdout/stdout_%J.log
#SBATCH --error=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs/%x/stderr/stderr_%J.log

# Number of GPUs to allocate (don't forget to select a partition with GPUs)
#SBATCH --partition=accel_ai
#SBATCH --gres=gpu:2
### SBATCH -t 0-00:00

# Number of CPU cores per task to allocate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

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

nohup mlflow server --host localhost --port 6006 --backend-store-uri file:/scratch/c.c21051562/workspace/arrg_img2text/outputs/mlruns > /dev/null 2>&1 &
echo "MLflow server started"

echo "Running script ... (job: $SLURM_JOB_NAME $SLURM_JOB_ID)"
# export TORCH_DISTRIBUTED_DEBUG=INFO
accelerate launch \
    --multi_gpu \
    --main_process_port 29555 \
    /scratch/c.c21051562/workspace/arrg_img2text/1_cls_effu_notallimg_fast.py \
    --from_bash \
    --config_file /scratch/c.c21051562/workspace/arrg_img2text/config/sunbird/1_imgcls_notallimg_fast.yaml \
    --output_name $SLURM_JOB_NAME \
    --jobid $SLURM_JOB_ID \
    # --resume_from_checkpoint
echo "Script finished."

python /scratch/c.c21051562/workspace/test_email.py --from_bash --subject "Sunbird Done: $SLURM_JOB_NAME"

# 查找所有运行中的 MLflow 进程
pids=$(ps aux | grep '[m]lflow' | awk '{print $2}')
echo "Killing MLflow server processes: $pids" 
if [ -z "$pids" ]; then
  echo "No MLflow processes found."
else
  for pid in $pids; do
    kill $pid
    echo "Stopped process with PID: $pid"
  done
fi
  

# sbatch /scratch/c.c21051562/workspace/arrg_img2text/run_sunbird.sh
# scontrol show job JOBID
# scontrol show job JOBID | grep NodeList
# scancel JOBID

# tensorboard --logdir=/scratch/c.c21051562/workspace/arrg_img2text/outputs/logs --port=6006

# Check process and kill
# ps aux | grep <进程名>
# kill <PID>

# ssh -L 6007:localhost:6006 yuxiang@10.97.37.97
# ssh -L 6007:localhost:6006 -J c.c21051562@sunbird.swansea.ac.uk c.c21051562@ccs2111

# ssh -L 6007:localhost:6006 c.c21051562@sunbird.swansea.ac.uk
# cd /scratch/c.c21051562/workspace/arrg_img2text/outputs/
# conda activate arrg_img2text
# mlflow server --host 127.0.0.1 --port 6006

