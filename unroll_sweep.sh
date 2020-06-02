#!/bin/bash

num_vals=1000000
window_size=4000
unroll_factors=(1 2 4 8)
for unroll_factor in "${unroll_factors[@]}"
do
   echo "./build/test_cuda_api_wrapper --num_vals ${num_vals} --window_size ${window_size} -k ${unroll_factor}"
   ./build/test_cuda_api_wrapper --num_vals ${num_vals} --window_size ${window_size} -k ${unroll_factor}
done
