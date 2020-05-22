//#include <cub.cuh>

#include "sliding_window.cuh"

my_float2 get_rand_my_float2( void ) {
   static std::default_random_engine r_engine;
   static std::uniform_real_distribution<> udist(0, 1); // range 0 - 1

   return {(float)udist(r_engine),(float)udist(r_engine)};
}

int my_test( const int window_size, const int num_vals, 
      const bool debug = false ) {

   debug_printf( debug, "%s(): Sliding Window Avg of %d elements\n", 
      __func__, num_vals ); 

   auto h_vals = std::unique_ptr<my_float2>(new my_float2[num_vals]);
   auto h_results = std::unique_ptr<my_float2>(new my_float2[num_vals]);

   std::cout << "Vals: ";
   for( int index = 0; index < num_vals; ++index ) {
      h_vals.get()[index] = get_rand_my_float2();
      /*std::cout << "{" << h_vals.get()[index].x << ", "*/
         /*<< h_vals.get()[index].y << "} ";*/
      std::cout << h_vals.get()[index] << " ";
   } 
   std::cout << "\n"; 
   return SUCCESS;
}
