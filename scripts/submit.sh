#!/bin/bash
MLFLOW_START=6123
MAIN_START=29123
JOB_NAME="7_3_text"

target_script=/home/yuxiang/liao/workspace/arrg_img2text/scripts/test.sh

# 对于findings来说，只考虑以下30个即可
# ['lung', 'lungs', 'heart', 'lobe', 'lining', 'contour', 'base', 'thorax', 'parenchyma', 'lower lobe of left lung', 'hemidiaphragm', 'carina', 'aorta', 'rib', 'caudomedial auditory cortex', 'base of prostate', 'stomach', 'mediastinum', 'spine', 'vasculature', 'airspace', 'lung zone', 'costophrenic sulcus', 'wall', 'hemithorax', 'collateral trigone of lateral ventricle', 'portion of tissue', 'right atrium', 'lower lobe of right lung', 'fluid', 'middle', 'chest wall', 'superior vena cava', 'upper lobe of right lung', 'thoracic vertebral column', 'vein', 'diaphragm', 'hilum', 'upper extremity', 'limb of body', 'paranasal sinuses', 'thoracic aorta', 'apex of prostate']

pairs=(
  "lung|['lung']"
  "lungs|['lungs']"
  "heart|['heart']"
  "lobe|['lobe']"
  "lining|['lining']"
  "contour|['contour']"
  "base|['base']"
  "thorax|['thorax']"
  "parenchyma|['parenchyma']"
  "lowerLobeLeftLung|['lower lobe of left lung']"
  "hemidiaphragm|['hemidiaphragm']"
  "carina|['carina']"
  "aorta|['aorta']"
  "rib|['rib']"
  "caudomedialAuditoryCortex|['caudomedial auditory cortex']"
  "baseProstate|['base of prostate']"
  "stomach|['stomach']"
  "mediastinum|['mediastinum']"
  "spine|['spine']"
  "vasculature|['vasculature']"
  "airspace|['airspace']"
  "lungZone|['lung zone']"
  "costophrenicSulcus|['costophrenic sulcus']"
  "wall|['wall']"
  "hemithorax|['hemithorax']"
  "collateralTrigoneLateralVentricle|['collateral trigone of lateral ventricle']"
  "portionTissue|['portion of tissue']"
  "rightAtrium|['right atrium']"
  "lowerLobeRightLung|['lower lobe of right lung']"
  "fluid|['fluid']"
  "middle|['middle']"
  "chestWall|['chest wall']"
  "superiorVenaCava|['superior vena cava']"
  "upperLobeRightLung|['upper lobe of right lung']"
  "thoracicVertebralColumn|['thoracic vertebral column']"
  "vein|['vein']"
  "diaphragm|['diaphragm']"
  "hilum|['hilum']"
  "upperExtremity|['upper extremity']"
  "limbBody|['limb of body']"
  "paranasalSinuses|['paranasal sinuses']"
  "thoracicAorta|['thoracic aorta']"
  "apexProstate|['apex of prostate']"
)
    

i=0
for pair in "${pairs[@]}"; do
  IFS="|" read -r job_name_suffix target_obs <<< "$pair"

  MLFLOW_PORT=$((MLFLOW_START + i))
  MAIN_PORT=$((MAIN_START + i))

  "$target_script" "$MLFLOW_PORT" "$MAIN_PORT" "$target_obs"

  i=$((i + 1))
done

# 需要 chmod +x both_script_path