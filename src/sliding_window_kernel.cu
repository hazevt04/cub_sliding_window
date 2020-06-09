#include "sliding_window_kernel.cuh"

//////////////////////////////////////
// The sliding window kernel
// Calculate sliding window average
//////////////////////////////////////
__global__ void sliding_window( float2* __restrict__ results, float2* const __restrict__ vals,
   const int window_size, const int num_results ) {

   int global_index = 1*(blockIdx.x * blockDim.x + threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   for (int index = global_index; index < num_results; index+=stride) {
      float2 t_val1 = {0.0, 0.0};
      for (int w_index = 0; w_index < window_size; ++w_index) {
         t_val1 = float2_add( t_val1, vals[index + w_index] );
      }
      t_val1 = float2_division_scalar( t_val1, (float)window_size );
      results[index] = t_val1;
   }
}

__global__ void sliding_window_unrolled_2x_inner( float2* __restrict__ results, float2* const __restrict__ vals,
   const int window_size, const int num_results ) {

   int global_index = 2*(blockIdx.x * blockDim.x + threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   for (int index = global_index; index < num_results; index+=stride) {
      float2 t_val1 = {0.0, 0.0};
      float2 t_val2 = {0.0, 0.0};
      for (int w_index = 0; w_index < window_size; w_index+=2) {
         // Windows for t_val1 and t_val2 overlap
         t_val1 = float2_add( t_val1, vals[index + w_index] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 1] );

         t_val2 = float2_add( t_val2, vals[index + w_index + 1] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 2] );
      }
      t_val1 = float2_division_scalar( t_val1, (float)window_size );
      t_val2 = float2_division_scalar( t_val2, (float)window_size );

      results[index] = t_val1;
      results[index+1] = t_val2;
   }
}


__global__ void sliding_window_unrolled_4x_inner( float2* __restrict__ results, float2* const __restrict__ vals,
   const int window_size, const int num_results ) {

   // Assuming one stream
   int global_index = 4*(blockIdx.x * blockDim.x + threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   for (int index = global_index; index < num_results; index+=stride) {
      float2 t_val1 = {0.0, 0.0};
      float2 t_val2 = {0.0, 0.0};
      float2 t_val3 = {0.0, 0.0};
      float2 t_val4 = {0.0, 0.0};
      for (int w_index = 0; w_index < window_size; w_index+=4) {
         t_val1 = float2_add( t_val1, vals[index + w_index] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 1] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 2] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 3] );
         
         t_val2 = float2_add( t_val2, vals[index + w_index + 1] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 2] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 3] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 4] );

         t_val3 = float2_add( t_val3, vals[index + w_index + 2] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 3] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 4] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 5] );

         t_val4 = float2_add( t_val4, vals[index + w_index + 3] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 4] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 5] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 6] );

      }
      t_val1 = float2_division_scalar( t_val1, (float)window_size );
      t_val2 = float2_division_scalar( t_val2, (float)window_size );
      t_val3 = float2_division_scalar( t_val3, (float)window_size );
      t_val4 = float2_division_scalar( t_val4, (float)window_size );

      results[index] = t_val1;
      results[index+1] = t_val2;
      results[index+2] = t_val3;
      results[index+3] = t_val4;
   }
}


__global__ void sliding_window_unrolled_8x_inner( float2* __restrict__ results, float2* const __restrict__ vals,
   const int window_size, const int num_results ) {

   // Assuming one stream
   int global_index = 8*(blockIdx.x * blockDim.x + threadIdx.x);
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   for (int index = global_index; index < num_results; index+=stride) {
      float2 t_val1 = {0.0, 0.0};
      float2 t_val2 = {0.0, 0.0};
      float2 t_val3 = {0.0, 0.0};
      float2 t_val4 = {0.0, 0.0};
      float2 t_val5 = {0.0, 0.0};
      float2 t_val6 = {0.0, 0.0};
      float2 t_val7 = {0.0, 0.0};
      float2 t_val8 = {0.0, 0.0};
      for (int w_index = 0; w_index < window_size; w_index+=8) {
         t_val1 = float2_add( t_val1, vals[index + w_index + 0] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 1] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 2] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 3] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 4] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 5] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 6] );
         t_val1 = float2_add( t_val1, vals[index + w_index + 7] );

         t_val2 = float2_add( t_val2, vals[index + w_index + 1] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 2] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 3] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 4] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 5] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 6] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 7] );
         t_val2 = float2_add( t_val2, vals[index + w_index + 8] );

         t_val3 = float2_add( t_val3, vals[index + w_index + 2] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 3] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 4] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 5] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 6] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 7] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 8] );
         t_val3 = float2_add( t_val3, vals[index + w_index + 9] );

         t_val4 = float2_add( t_val4, vals[index + w_index + 3] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 4] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 5] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 6] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 7] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 8] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 9] );
         t_val4 = float2_add( t_val4, vals[index + w_index + 10] );

         t_val5 = float2_add( t_val5, vals[index + w_index + 4] );
         t_val5 = float2_add( t_val5, vals[index + w_index + 5] );
         t_val5 = float2_add( t_val5, vals[index + w_index + 6] );
         t_val5 = float2_add( t_val5, vals[index + w_index + 7] );
         t_val5 = float2_add( t_val5, vals[index + w_index + 8] );
         t_val5 = float2_add( t_val5, vals[index + w_index + 9] );
         t_val5 = float2_add( t_val5, vals[index + w_index + 10] );
         t_val5 = float2_add( t_val5, vals[index + w_index + 11] );

         t_val6 = float2_add( t_val6, vals[index + w_index + 5] );
         t_val6 = float2_add( t_val6, vals[index + w_index + 6] );
         t_val6 = float2_add( t_val6, vals[index + w_index + 7] );
         t_val6 = float2_add( t_val6, vals[index + w_index + 8] );
         t_val6 = float2_add( t_val6, vals[index + w_index + 9] );
         t_val6 = float2_add( t_val6, vals[index + w_index + 10] );
         t_val6 = float2_add( t_val6, vals[index + w_index + 11] );
         t_val6 = float2_add( t_val6, vals[index + w_index + 12] );

         t_val7 = float2_add( t_val7, vals[index + w_index + 6] );
         t_val7 = float2_add( t_val7, vals[index + w_index + 7] );
         t_val7 = float2_add( t_val7, vals[index + w_index + 8] );
         t_val7 = float2_add( t_val7, vals[index + w_index + 9] );
         t_val7 = float2_add( t_val7, vals[index + w_index + 10] );
         t_val7 = float2_add( t_val7, vals[index + w_index + 11] );
         t_val7 = float2_add( t_val7, vals[index + w_index + 12] );
         t_val7 = float2_add( t_val7, vals[index + w_index + 13] );

         t_val8 = float2_add( t_val8, vals[index + w_index + 7] );
         t_val8 = float2_add( t_val8, vals[index + w_index + 8] );
         t_val8 = float2_add( t_val8, vals[index + w_index + 9] );
         t_val8 = float2_add( t_val8, vals[index + w_index + 10] );
         t_val8 = float2_add( t_val8, vals[index + w_index + 11] );
         t_val8 = float2_add( t_val8, vals[index + w_index + 12] );
         t_val8 = float2_add( t_val8, vals[index + w_index + 13] );
         t_val8 = float2_add( t_val8, vals[index + w_index + 14] );

      }
      t_val1 = float2_division_scalar( t_val1, (float)window_size );
      t_val2 = float2_division_scalar( t_val2, (float)window_size );
      t_val3 = float2_division_scalar( t_val3, (float)window_size );
      t_val4 = float2_division_scalar( t_val4, (float)window_size );
      t_val5 = float2_division_scalar( t_val5, (float)window_size );
      t_val6 = float2_division_scalar( t_val6, (float)window_size );
      t_val7 = float2_division_scalar( t_val7, (float)window_size );
      t_val8 = float2_division_scalar( t_val8, (float)window_size );

      results[index + 0] = t_val1;
      results[index + 1] = t_val2;
      results[index + 2] = t_val3;
      results[index + 3] = t_val4;
      results[index + 4] = t_val5;
      results[index + 5] = t_val6;
      results[index + 6] = t_val7;
      results[index + 7] = t_val8;
   }
}



