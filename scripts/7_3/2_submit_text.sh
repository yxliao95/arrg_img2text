#!/bin/bash
MLFLOW_START=6123
MAIN_START=29123
JOB_NAME="7_3_text"

target_script=/scratch/c.c21051562/workspace/arrg_img2text/scripts/7_3/1_run_7_3_text.sh

# 对于findings来说，只考虑以下30个即可
# ["effusion", "pneumothorax", "normal", "consolidation", "opacity", "clear", "atelectasis", "edema", "tube", "catheter", "pneumonia", "infiltrate", "pathophysiologic finding", "congestion", "degeneration", "wire", "enlargement", "infection", "thickening", "fracture", "pacemaker", "scoliosis", "chronic obstructive pulmonary disease", "medical device", "calcifications", "emphysema", "surgical clip", "atherosclerosis", "calcification", "granuloma"]
pairs=(
    "effu|['effusion']"
    "pneu|['pneumothorax']"
    "norm|['normal']"
    "consolidation|['consolidation']"
    "opac|['opacity']"
    "clear|['clear']"
    "atelectasis|['atelectasis']"
    "edema|['edema']"
    "tube|['tube']"
    "catheter|['catheter']"
    "pneumonia|['pneumonia']"
    "infiltrate|['infiltrate']"
    "pathFinding|['pathophysiologic finding']"
    "congestion|['congestion']"
    "degeneration|['degeneration']"
    "wire|['wire']"
    "enlargement|['enlargement']"
    "infection|['infection']"
    "thickening|['thickening']"
    "fracture|['fracture']"
    "pacemaker|['pacemaker']"
    "scoliosis|['scoliosis']"
    "copd|['chronic obstructive pulmonary disease']"
    "medicalDevice|['medical device']"
    "calcifications|['calcifications']"
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

  sbatch --job-name="${JOB_NAME}_${i}_${job_name_suffix}" \
         "$target_script" "$MLFLOW_PORT" "$MAIN_PORT" "$target_obs"

  i=$((i + 1))
  sleep 1
done

# 需要 chmod +x both_script_path