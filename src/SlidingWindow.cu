#include "SlidingWindow.cuh"

SlidingWindow::SlidingWindow( int new_num_vals, int new_window_size, bool new_debug = false ) {
   num_vals = new_num_vals;
   window_size = new_window_size;
   num_results = num_vals - window_size;
   debug = new_debug;

   h_vals = std::unique_ptr<float2>(new float2[new_num_vals]);
   h_results = std::unique_ptr<float2>(new float2[num_results]);
   expected_results = std::unique_ptr<float2>(new float2[num_results]);

   auto current_device = cuda::device::current::get();
   d_vals = cuda::memory::device::make_unique<float2[]>(current_device, new_num_vals);
   d_results = cuda::memory::device::make_unique<float2[]>(current_device, num_results);
   
}

// Move Constructor
SlidingWindow::SlidingWindow( SlidingWindow&& other ) noexcept {
   num_vals = other.num_vals;
   window_size = other.window_size;
   num_results = other.num_results;
   debug = other.debug;

   h_vals = std::move( other.h_vals );
   h_results = std::move(other.h_results);
   expected_results = std::move(other.expected_results);
   
   d_vals = std::move( other.d_vals );
   d_results = std::move( other.d_results );

   other.num_vals = 0;
   other.window_size = 0;
   other.num_results = 0;

   other.h_vals.reset();
   other.h_results = nullptr;
   other.expected_results = nullptr;

   other.d_vals = nullptr;
   other.d_results = nullptr;
   
}


// Move Assignemt constructor
SlidingWindow& SlidingWindow::operator=( SlidingWindow&& other ) noexcept {
   if ( this != &other ) {
      num_vals = other.num_vals;
      window_size = other.window_size;
      num_results = other.num_results;
      debug = other.debug;

      h_vals = std::move( other.h_vals );
      h_results = std::move( other.h_results );
      expected_results = std::move( other.expected_results );

      d_vals = std::move( other.d_vals );
      d_results = std::move( other.d_results );
      
      other.h_vals.reset();
      other.h_results.reset();
      other.expected_results.reset();

      other.num_vals = 0;
      other.window_size = 0;
      other.num_results = 0;
   }
   return *this;
}

SlidingWindow::~SlidingWindow() {

   h_vals.reset();
   h_results.reset();
   expected_results.reset();

   d_vals.reset();
   d_results.reset();
   num_vals = 0;
   num_results = 0;
   window_size = 0;
}


void SlidingWindow::gen_vals() {
   for( int index = 0; index < num_vals; ++index ) {
      h_vals.get()[index].x = (float)index;
      h_vals.get()[index].y = (float)index;
   }   
}


void SlidingWindow::gen_expected() {
   
   expected_results.get()[0].x = 0.f;
   expected_results.get()[0].y = 0.f;

   for( int s_index = 0; s_index < window_size; ++s_index ) {
      expected_results.get()[0].x += h_vals.get()[s_index].x;
      expected_results.get()[0].y += h_vals.get()[s_index].y;
   }
   debug_printf( debug, "%s(): expected_results.get()[0].= %f, %f\n", __func__, 
      expected_results.get()[0].x, expected_results.get()[0].y );    

   float prev_x_sum = expected_results.get()[0].x;
   float prev_y_sum = expected_results.get()[0].y;

   for ( int index = 1; index < num_results; ++index ) {
      expected_results.get()[index].x = prev_x_sum - h_vals.get()[index-1].x + h_vals.get()[index + window_size - 1].x;
      expected_results.get()[index].y = prev_y_sum - h_vals.get()[index-1].y + h_vals.get()[index + window_size - 1].y;
      prev_x_sum = expected_results.get()[index].x;
      prev_y_sum = expected_results.get()[index].y;
   }
}


void SlidingWindow::run() {

   gen_vals();
   gen_expected();

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

   Time_Point start;
   Duration_ms duration_ms;
   
   size_t size_vals = num_vals * sizeof(float2);
   size_t size_results = num_results * sizeof(float2);

   gen_vals();
   gen_expected();
   
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

   cuda::memory::copy( h_results.get(), d_results.get(), size_results );
   
   duration_ms = Steady_Clock::now() - start;
   debug_printf( true, "%d vals, %d window size, %f ms\n", num_vals, window_size, duration_ms.count() );

   if ( debug ) {
      std::cout << "Results: \n";
      for( int index = 0; index < num_results; ++index ) {
         std::cout << "[" << index << "]: {" << h_results.get()[index].x << ", " 
            << h_results.get()[index].y << "}\n";
      } 
      std::cout << "\n"; 
   }   

}
