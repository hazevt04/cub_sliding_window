//#include <cub.cuh>

#include "sliding_window.cuh"

my_float2 get_rand_my_float2( void ) {
   static std::default_random_engine r_engine;
   static std::uniform_real_distribution<> udist(0, 1); // range 0 - 1

   return {(float)udist(r_engine),(float)udist(r_engine)};
}

float2 get_rand_float2( void ) {
   static std::default_random_engine r_engine;
   static std::uniform_real_distribution<> udist(0, 1); // range 0 - 1

   float2 result{(float)udist(r_engine),(float)udist(r_engine)};
   return result;
}

int my_test( const int window_size, const int num_vals, 
      const bool debug = false ) {

   debug_printf( debug, "%s(): Sliding Window Avg of %d elements\n", 
      __func__, num_vals ); 

   size_t size_vals = num_vals * sizeof(float2);
   auto h_vals = std::unique_ptr<float2>(new float2[num_vals]);
   auto h_results = std::unique_ptr<float2>(new float2[num_vals]);

   std::cout << "Num Vals = " << num_vals << "\n";
   std::cout << "Window Size = " << window_size << "\n";
   std::cout << "Vals: ";
   for( int index = 0; index < num_vals; ++index ) {
      h_vals.get()[index] = get_rand_float2();
      std::cout << "{" << h_vals.get()[index].x << ", ";
      std::cout << h_vals.get()[index].y << "} ";
   } 
   std::cout << "\n"; 

   auto current_device = cuda::device::current::get();

   auto d_vals = cuda::memory::device::make_unique<float2[]>(current_device, num_vals);
   auto d_results = cuda::memory::device::make_unique<float2[]>(current_device, num_vals);

   cuda::memory::copy( d_vals.get(), h_vals.get(), size_vals );

   int threads_per_block = 256;
   int blocks_per_grid = CEILING( num_vals, threads_per_block );
   
   std::cout << "CUDA kernel launch with " << blocks_per_grid
      << " blocks of " << threads_per_block << " threads\n";  
   
   cuda::grid::dimensions_t grid_dims( blocks_per_grid, 1, 1 );
   cuda::grid::dimensions_t block_dims( threads_per_block, 1, 1 );
   cuda::memory::shared::size_t s_mem_size( 0u );
   cuda::launch_configuration_t launch_configuration = cuda::make_launch_config( grid_dims, block_dims, s_mem_size );
   
   cuda::launch(
      sliding_window,
      //cuda::launch_configuration_t( blocks_per_grid, threads_per_block ),
      launch_configuration,
      d_results.get(), d_vals.get(), window_size, num_vals
   );

   cuda::memory::copy( h_results.get(), d_results.get(), size_vals );

   std::cout << "Results: ";
   for( int index = 0; index < num_vals; ++index ) {
      std::cout << "{" << h_results.get()[index].x << ", ";
      std::cout << h_results.get()[index].y << "} ";
   } 
   std::cout << "\n"; 

   return SUCCESS;
}
