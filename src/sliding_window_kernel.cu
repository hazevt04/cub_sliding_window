#include "sliding_window_kernel.cuh"

//////////////////////////////////////
// THE Kernel (sliding window)
// Calculate sliding window average
//////////////////////////////////////
__global__ void sliding_window( float2* __restrict__ results, float2* const __restrict__ vals, 
   const int window_size, const int num_results ) {
   
   // Assuming one stream
   int global_index = 1*(blockIdx.x * blockDim.x) + 1*(threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   for (int index = global_index; index < num_results; index+=stride) {
      float2 t_val = {0.0, 0.0};
      for (int w_index = 0; w_index < window_size; w_index++) {
         ADD_COMPLEX( t_val, t_val, vals[index + w_index] );      
      }
      DIVIDE_COMPLEX_BY_SCALAR( t_val, t_val, (float)window_size );

      ASSIGN_COMPLEX( results[index], t_val );
   }
}
