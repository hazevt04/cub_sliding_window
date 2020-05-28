#!/bin/bash

num_vals=(256 512 1024 2048 4098 8192 16384 32768 65536 131072 262144 524288 1048576)
window_sizes=(64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144)
for num_val in "${num_vals[@]}"
do
   echo "./build/test_cuda_api_wrapper --num_vals ${num_val} --window_size $(($num_val/4))"
   ./build/test_cuda_api_wrapper --num_vals ${num_val} --window_size $(($num_val/4))
done
