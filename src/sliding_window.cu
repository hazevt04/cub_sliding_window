//#include <cub.cuh>

#include "sliding_window.cuh"

float2 get_next_float2( void ) {
   static float2 result{0.0, 0.0};

   result.x++;
   result.y++;
   return result;
}

void gen_float2s( float2* vals, const int num_vals ) {

   for( int index = 0; index < num_vals; ++index ) {
      vals[index].x = (float)index;
      vals[index].y = (float)index;
   }
}


void gen_expected( float2* expected_results, const float2* vals, 
   const int num_results, const int window_size, const bool debug = false ) {
   
   expected_results[0].x = 0.f;
   expected_results[0].y = 0.f;

   for( int s_index = 0; s_index < window_size; ++s_index ) {
      expected_results[0].x += vals[s_index].x;
      expected_results[0].y += vals[s_index].y;
   }
   debug_printf( debug, "%s(): expected_results[0] = %f, %f\n", __func__, 
      expected_results[0].x, expected_results[0].y );    

   float prev_x_sum = expected_results[0].x;
   float prev_y_sum = expected_results[0].y;

   for ( int index = 1; index < num_results; ++index ) {
      expected_results[index].x = prev_x_sum - vals[index-1].x + vals[index + window_size - 1].x;
      expected_results[index].y = prev_y_sum - vals[index-1].y + vals[index + window_size - 1].y;
      prev_x_sum = expected_results[index].x;
      prev_y_sum = expected_results[index].y;
   }
}


float2 get_rand_float2( void ) {
   static std::default_random_engine r_engine;
   static std::uniform_real_distribution<> udist(0, 1); // range 0 - 1

   float2 result{(float)udist(r_engine),(float)udist(r_engine)};
   return result;
}


int run_kernel( const int window_size, const int num_vals, 
      const bool debug = false ) {

   debug_printf( debug, "%s(): Sliding Window Avg of %d elements\n", 
      __func__, num_vals ); 

   int num_results = num_vals - window_size;
   size_t size_vals = num_vals * sizeof(float2);
   size_t size_vals_windowed = num_results * sizeof(float2);
   
   auto current_device = cuda::device::current::get();

   auto h_vals = cuda::memory::host::make_unique<float2[]>(num_vals);
   auto h_results = cuda::memory::host::make_unique<float2[]>(num_results);
   auto expected_results = cuda::memory::host::make_unique<float2[]>(num_results);

   //std::generate(h_vals.get(), h_vals.get() + num_vals, get_rand_float2 ); 
   //std::generate(h_vals.get(), h_vals.get() + num_vals, get_next_float2 ); 
   gen_float2s( h_vals.get(), num_vals );
   gen_expected( expected_results.get(), h_vals.get(), num_results, window_size, debug );

   if ( debug ) {
      std::cout << "Num Vals = " << num_vals << "\n";
      std::cout << "Window Size = " << window_size << "\n";
      std::cout << "Num Vals Windowed = " << num_results << "\n";
      std::cout << "Vals: \n";
      for( int index = 0; index < num_vals; ++index ) {
         std::cout << "[" << index << "] {" 
            << h_vals.get()[index].x << ", " << h_vals.get()[index].y << "}\n ";
      } 
      std::cout << "\n"; 
   }
   
   int threads_per_block = 1024;
   int blocks_per_grid = CEILING( num_results, threads_per_block );
   
   cuda::grid::dimensions_t grid_dims( blocks_per_grid, 1, 1 );
   cuda::grid::dimensions_t block_dims( threads_per_block, 1, 1 );
   cuda::memory::shared::size_t s_mem_size( 0u );
   cuda::launch_configuration_t launch_configuration = cuda::make_launch_config( grid_dims, 
         block_dims, s_mem_size );
   
   debug_printf( debug, "blocks_per_grid = %d\n", blocks_per_grid );
   debug_printf( debug, "threads_per_block = %d\n",threads_per_block );
   debug_printf( debug, "total threads = block_per_grid x threads_per_block = %d\n",
      ( threads_per_block * blocks_per_grid ) );

   auto d_vals = cuda::memory::device::make_unique<float2[]>(current_device, num_vals);
   auto d_results = cuda::memory::device::make_unique<float2[]>(current_device, num_results);

   Time_Point start;
   Duration_ms gpu_ms;

   start = Steady_Clock::now();
   cuda::memory::copy( d_vals.get(), h_vals.get(), size_vals );
   
   if ( debug ) {
      std::cout << "CUDA kernel launch with " << blocks_per_grid
         << " blocks of " << threads_per_block << " threads\n";
   }

   cuda::launch(
      sliding_window,
      launch_configuration,
      d_results.get(), d_vals.get(), window_size, num_results
   );

   cuda::memory::copy( h_results.get(), d_results.get(), size_vals_windowed );

   gpu_ms = Steady_Clock::now() - start;
   debug_printf( true, "%d vals, %d window size, %f ms\n", num_vals, window_size, gpu_ms.count() );

   if ( debug ) {
      std::cout << "Results: \n";
      for( int index = 0; index < num_results; ++index ) {
         std::cout << "[" << index << "]: {" << h_results.get()[index].x << ", " 
            << h_results.get()[index].y << "}\n";
      } 
      std::cout << "\n"; 
   }
   return SUCCESS;

}
