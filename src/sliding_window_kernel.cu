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
      float2 t_val = {0.0, 0.0};
      for (int w_index = 0; w_index < window_size; ++w_index) {
         ADD_COMPLEX( t_val, t_val, vals[index + w_index] );
      }
      DIVIDE_COMPLEX_BY_SCALAR( t_val, t_val, (float)window_size );

      ASSIGN_COMPLEX( results[index], t_val );
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
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 1] );

         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 1] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 2] );
      }
      DIVIDE_COMPLEX_BY_SCALAR( t_val1, t_val1, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val2, t_val2, (float)window_size );

      ASSIGN_COMPLEX( results[index], t_val1 );
      ASSIGN_COMPLEX( results[index+1], t_val2 );
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
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 1] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 2] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 3] );
         
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 1] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 2] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 3] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 4] );

         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 2] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 3] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 4] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 5] );

         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 3] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 4] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 5] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 6] );

      }
      DIVIDE_COMPLEX_BY_SCALAR( t_val1, t_val1, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val2, t_val2, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val3, t_val3, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val4, t_val4, (float)window_size );

      ASSIGN_COMPLEX( results[index], t_val1 );
      ASSIGN_COMPLEX( results[index+1], t_val2 );
      ASSIGN_COMPLEX( results[index+2], t_val3 );
      ASSIGN_COMPLEX( results[index+3], t_val4 );
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
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 0] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 1] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 2] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 3] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 4] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 5] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 6] );
         ADD_COMPLEX( t_val1, t_val1, vals[index + w_index + 7] );

         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 1] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 2] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 3] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 4] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 5] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 6] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 7] );
         ADD_COMPLEX( t_val2, t_val2, vals[index + w_index + 8] );

         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 2] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 3] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 4] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 5] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 6] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 7] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 8] );
         ADD_COMPLEX( t_val3, t_val3, vals[index + w_index + 9] );

         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 3] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 4] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 5] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 6] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 7] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 8] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 9] );
         ADD_COMPLEX( t_val4, t_val4, vals[index + w_index + 10] );

         ADD_COMPLEX( t_val5, t_val5, vals[index + w_index + 4] );
         ADD_COMPLEX( t_val5, t_val5, vals[index + w_index + 5] );
         ADD_COMPLEX( t_val5, t_val5, vals[index + w_index + 6] );
         ADD_COMPLEX( t_val5, t_val5, vals[index + w_index + 7] );
         ADD_COMPLEX( t_val5, t_val5, vals[index + w_index + 8] );
         ADD_COMPLEX( t_val5, t_val5, vals[index + w_index + 9] );
         ADD_COMPLEX( t_val5, t_val5, vals[index + w_index + 10] );
         ADD_COMPLEX( t_val5, t_val5, vals[index + w_index + 11] );

         ADD_COMPLEX( t_val6, t_val6, vals[index + w_index + 5] );
         ADD_COMPLEX( t_val6, t_val6, vals[index + w_index + 6] );
         ADD_COMPLEX( t_val6, t_val6, vals[index + w_index + 7] );
         ADD_COMPLEX( t_val6, t_val6, vals[index + w_index + 8] );
         ADD_COMPLEX( t_val6, t_val6, vals[index + w_index + 9] );
         ADD_COMPLEX( t_val6, t_val6, vals[index + w_index + 10] );
         ADD_COMPLEX( t_val6, t_val6, vals[index + w_index + 11] );
         ADD_COMPLEX( t_val6, t_val6, vals[index + w_index + 12] );

         ADD_COMPLEX( t_val7, t_val7, vals[index + w_index + 5] );
         ADD_COMPLEX( t_val7, t_val7, vals[index + w_index + 6] );
         ADD_COMPLEX( t_val7, t_val7, vals[index + w_index + 7] );
         ADD_COMPLEX( t_val7, t_val7, vals[index + w_index + 8] );
         ADD_COMPLEX( t_val7, t_val7, vals[index + w_index + 9] );
         ADD_COMPLEX( t_val7, t_val7, vals[index + w_index + 10] );
         ADD_COMPLEX( t_val7, t_val7, vals[index + w_index + 11] );
         ADD_COMPLEX( t_val7, t_val7, vals[index + w_index + 12] );

         ADD_COMPLEX( t_val8, t_val8, vals[index + w_index + 6] );
         ADD_COMPLEX( t_val8, t_val8, vals[index + w_index + 7] );
         ADD_COMPLEX( t_val8, t_val8, vals[index + w_index + 8] );
         ADD_COMPLEX( t_val8, t_val8, vals[index + w_index + 9] );
         ADD_COMPLEX( t_val8, t_val8, vals[index + w_index + 10] );
         ADD_COMPLEX( t_val8, t_val8, vals[index + w_index + 11] );
         ADD_COMPLEX( t_val8, t_val8, vals[index + w_index + 12] );
         ADD_COMPLEX( t_val8, t_val8, vals[index + w_index + 13] );

      }
      DIVIDE_COMPLEX_BY_SCALAR( t_val1, t_val1, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val2, t_val2, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val3, t_val3, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val4, t_val4, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val5, t_val5, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val6, t_val6, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val7, t_val7, (float)window_size );
      DIVIDE_COMPLEX_BY_SCALAR( t_val8, t_val8, (float)window_size );

      ASSIGN_COMPLEX( results[index + 0], t_val1 );
      ASSIGN_COMPLEX( results[index + 1], t_val2 );
      ASSIGN_COMPLEX( results[index + 2], t_val3 );
      ASSIGN_COMPLEX( results[index + 3], t_val4 );
      ASSIGN_COMPLEX( results[index + 4], t_val5 );
      ASSIGN_COMPLEX( results[index + 5], t_val6 );
      ASSIGN_COMPLEX( results[index + 6], t_val7 );
      ASSIGN_COMPLEX( results[index + 7], t_val8 );
   }
}


