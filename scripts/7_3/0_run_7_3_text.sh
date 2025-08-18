#!/bin/bash

#SBATCH --job-name=7_3_text_
#SBATCH --account=scw2258

# Job stdout file. The '%J' = job number. %x = job name
#SBATCH --output=/scratch/c.c21051562/workspace/arrg_img2text/outputs_7_3_mix/logs/%x/stdout/stdout_%J.log
#SBATCH --error=/scratch/c.c21051562/workspace/arrg_img2text/outputs_7_3_mix/logs/%x/stderr/stderr_%J.log

# Number of GPUs to allocate (don't forget to select a partition with GPUs)
#SBATCH --partition=accel_ai
#SBATCH --gres=gpu:2
### SBATCH -t 0-00:00

# Number of CPU cores per task to allocate, (maximun of 4 cpus for 1 gpu, 16 for compute nodes)
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8

mlflow_port=${1}
main_process_port=${2}
target_observation=${3}
num_epoch=${4}

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

python /scratch/c.c21051562/workspace/test_email.py --from_bash --subject "【Sunbird Start】: $SLURM_JOB_NAME"

nohup mlflow server --host localhost --port $mlflow_port --backend-store-uri file:/scratch/c.c21051562/workspace/arrg_img2text/outputs_7_3_mix/mlruns > /dev/null 2>&1 &
echo "MLflow server started"


echo "Running script ... (job: $SLURM_JOB_NAME $SLURM_JOB_ID)"
export TORCH_DISTRIBUTED_DEBUG=OFF # OFF, INFO, or DETAIL
export NCCL_TIMEOUT=1800  # 默认是 1800 秒（30 分钟），你可以设置更大，比如 3600
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 避免碎片化

accelerate launch\
    --multi_gpu \
    --num_processes 2 \
    --main_process_port $main_process_port \
    /scratch/c.c21051562/workspace/arrg_img2text/7_3_not_clsssify_not_inject.py \
    --from_bash \
    --config_file /scratch/c.c21051562/workspace/arrg_img2text/config/sunbird/7_3_mix.yaml \
    --output_name $SLURM_JOB_NAME \
    --jobid $SLURM_JOB_ID \
    --mlflow_port $mlflow_port \
    --run_mode finetune \
    --target_observation "${target_observation}" \
    --num_epochs $num_epoch \
    # --resume_from_checkpoint

echo "Script [finetune] finished."

accelerate launch\
    --multi_gpu \
    --num_processes 2 \
    --main_process_port $main_process_port \
    /scratch/c.c21051562/workspace/arrg_img2text/7_3_not_clsssify_not_inject.py \
    --from_bash \
    --config_file /scratch/c.c21051562/workspace/arrg_img2text/config/sunbird/7_3_mix.yaml \
    --output_name $SLURM_JOB_NAME \
    --jobid $SLURM_JOB_ID \
    --mlflow_port $mlflow_port \
    --classification_only \
    --run_mode eval_finetuned \
    --target_observation "${target_observation}" \

echo "Script [eval_finetuned] finished."

python /scratch/c.c21051562/workspace/test_email.py --from_bash --subject "【Sunbird Done】: $SLURM_JOB_NAME"

# 查找运行在该端口的 mlflow 进程
pids=$(lsof -i :$mlflow_port -sTCP:LISTEN -t)
echo "Killing MLflow server on port $mlflow_port. PIDs: $pids"
if [ -z "$pids" ]; then
  echo "No MLflow processes found on port $mlflow_port."
else
  for pid in $pids; do
    kill $pid
    echo "Stopped process with PID: $pid"
  done
fi