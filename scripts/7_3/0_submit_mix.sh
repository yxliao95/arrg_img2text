#!/bin/bash
MLFLOW_START=5323
MAIN_START=24623
JOB_NAME="7_3_text_"

target_script=/scratch/c.c21051562/workspace/arrg_img2text/scripts/7_3/0_run_7_3_text.sh

# 对于findings来说，只考虑以下30个即可
# ["effusion", "pneumothorax", "normal", "consolidation", "opacity", "clear", "atelectasis", "edema", "tube", "catheter", "pneumonia", "infiltrate", "pathophysiologic finding", "congestion", "degeneration", "wire", "enlargement", "infection", "thickening", "fracture", "pacemaker", "scoliosis", "chronic obstructive pulmonary disease", "medical device", "calcifications", "emphysema", "surgical clip", "atherosclerosis", "calcification", "granuloma"]
pairs=(
    "thickening|['thickening']"
    "emphysema|['emphysema']"
    "surgicalClip|['surgical clip']"
    "atherosclerosis|['atherosclerosis']"
    "calcification|['calcification']"
    "granuloma|['granuloma']"
)

i=0
for pair in "${pairs[@]}"; do
  IFS="|" read -r job_name_suffix target_obs <<< "$pair"

  MLFLOW_PORT=$((MLFLOW_START + i))
  MAIN_PORT=$((MAIN_START + i))

  sbatch --job-name="${JOB_NAME}_${job_name_suffix}_epoch2" \
         "$target_script" "$MLFLOW_PORT" "$MAIN_PORT" "$target_obs" 2

  i=$((i + 1))
  sleep 1
done

for pair in "${pairs[@]}"; do
  IFS="|" read -r job_name_suffix target_obs <<< "$pair"

  MLFLOW_PORT=$((MLFLOW_START + i))
  MAIN_PORT=$((MAIN_START + i))

  sbatch --job-name="${JOB_NAME}_${job_name_suffix}_epoch5" \
         "$target_script" "$MLFLOW_PORT" "$MAIN_PORT" "$target_obs" 5

  i=$((i + 1))
  sleep 1
done

pairs=(
    "emphysema|['emphysema']"
    "surgicalClip|['surgical clip']"
    "atherosclerosis|['atherosclerosis']"
    "calcification|['calcification']"
)
for pair in "${pairs[@]}"; do
  IFS="|" read -r job_name_suffix target_obs <<< "$pair"

  MLFLOW_PORT=$((MLFLOW_START + i))
  MAIN_PORT=$((MAIN_START + i))

  sbatch --job-name="${JOB_NAME}_${job_name_suffix}_epoch10" \
         "$target_script" "$MLFLOW_PORT" "$MAIN_PORT" "$target_obs" 10

  i=$((i + 1))
  sleep 1
done

# 需要 chmod +x both_script_path