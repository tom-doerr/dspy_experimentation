#!/bin/bash

#python3 -m vllm.entrypoints.openai.api_server --model TheBloke/Xwin-LM-70B-V0.1-AWQ --quantization awq --dtype half --tensor-parallel-size 2 --port 8427 --gpu-memory-utilization 0.6 
#python3 -m vllm.entrypoints.openai.api_server --model TheBloke/Xwin-LM-70B-V0.1-AWQ --quantization awq --dtype half --tensor-parallel-size 2 --port 8427 --gpu-memory-utilization 1.0 --max-num-batched-tokens 20000

#python3 -m vllm.entrypoints.openai.api_server --model Xwin-LM/Xwin-LM-70B-V0.1 --dtype half --tensor-parallel-size 2 --port 8427 --gpu-memory-utilization 1.0 
#python3 -m vllm.entrypoints.openai.api_server --model TheBloke/Xwin-LM-70B-V0.1-AWQ --quantization awq --dtype half --tensor-parallel-size 2 --port 8427 --gpu-memory-utilization 0.6 --engine-use-ray
python3 -m vllm.entrypoints.openai.api_server --model TheBloke/Xwin-LM-70B-V0.1-AWQ --quantization awq --dtype half --tensor-parallel-size 2 --port 8427 --gpu-memory-utilization 0.6
#python3 -m vllm.entrypoints.openai.api_server --dtype half --tensor-parallel-size 2 --port 8427 --gpu-memory-utilization 0.6

