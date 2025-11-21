#!/bin/bash

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

environs="WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1"

inference_case_config="inference_case_configs/multi_prompts/rose2.yaml"

input_image=$(grep 'input_image:' ${inference_case_config} | sed 's/.*input_image:[[:space:]]*"\(.*\)"/\1/')

run_cmd="$environs python sample_video_i2v.py --base configs/cogvideox_5b_i2v.yaml configs/inference_i2v.yaml --custom-config $inference_case_config --input_image $input_image"
echo ${run_cmd}
eval ${run_cmd}

echo "DONE on $(hostname)"
