#!/bin/bash
set -e

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# single-GPU distributed environ bootstrap (same as your multi-prompt script)
environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

# ------------------------------------------------------
# Path to your I2V model config (the modified one)
# ------------------------------------------------------
i2v_model_config="configs/cogvideox_2b_contextKV.yaml"

# ------------------------------------------------------
# Minimal custom inference case config (like rose.yaml)
# We will create: inference_case_configs/i2v.yaml
# ------------------------------------------------------
i2v_case_config="inference_case_configs/i2v.yaml"

# ------------------------------------------------------
# Run inference the SAME WAY as CogVideoX multi-prompt
# ------------------------------------------------------
run_cmd="$environs python sample_video.py \
    --base $i2v_model_config configs/inference.yaml \
    --custom-config $i2v_case_config"

echo "[I2V] Running:"
echo "$run_cmd"
echo ""

eval ${run_cmd}

echo ""
echo "========================================="
echo " I2V Inference COMPLETE on $(hostname)"
echo "========================================="
