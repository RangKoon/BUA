#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 nohup ./venv_vllm/bin/vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000 --tensor-parallel-size 2 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.9 > vllm_base.log 2>&1 &
echo "PID: $!"
