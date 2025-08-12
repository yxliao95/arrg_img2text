#!/bin/bash

START=8083653
END=8083694

for JOBID in $(seq "$START" "$END"); do
  echo "Cancelling job $JOBID"
  scancel "$JOBID"
  sleep 1
done

# 需要 chmod +x both_script_path