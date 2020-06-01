#include "SlidingWindowConfig.cuh"
#include "SlidingWindow.cuh"

SlidingWindow::SlidingWindow( int new_num_vals, int new_window_size, 
   KernelSel new_kernel_sel, bool new_debug = false ) : num_vals( new_num_vals ),
      window_size( new_window_size ),
      kernel_sel( new_kernel_sel ),
      debug( new_debug ) {

   num_results = new_num_vals - new_window_size;
   size_t vals_size = new_num_vals * sizeof(float2);
   size_t results_size = num_results * sizeof(float2);

   auto current_device = cuda::device::current::get();
   // initial_visibility != initial_visibility_t::to_all_devices implies cudaMemAttachHost flag for CUDA host memory allocation
   vals = cuda::memory::managed::make_unique<float2[]>( vals_size,
        current_device.supports_concurrent_managed_access() ?
			cuda::memory::managed::initial_visibility_t::to_supporters_of_concurrent_managed_access:
			cuda::memory::managed::initial_visibility_t::to_all_devices );

   results = cuda::memory::managed::make_unique<float2[]>( results_size );
   expected_results = cuda::memory::managed::make_unique<float2[]>( results_size );

   /*auto current_device = cuda::device::current::get();*/
   /*vals = cuda::memory::device::make_unique<float2[]>(current_device, new_num_vals);*/
   /*results = cuda::memory::device::make_unique<float2[]>(current_device, num_results);*/
   
}


SlidingWindow::SlidingWindow( const SlidingWindowConfig& config ):
      num_vals( config.num_vals ),
      window_size( config.window_size ),
      kernel_sel( config.kernel_sel ),
      num_results( config.num_results ),
      debug( config.debug ) {
   
   size_t vals_size = config.num_vals * sizeof(float2);
   size_t results_size = config.num_results * sizeof(float2);
   
   auto current_device = cuda::device::current::get();

   // initial_visibility != initial_visibility_t::to_all_devices implies cudaMemAttachHost flag for CUDA host memory allocation
   vals = cuda::memory::managed::make_unique<float2[]>( vals_size,
        current_device.supports_concurrent_managed_access() ?
			cuda::memory::managed::initial_visibility_t::to_supporters_of_concurrent_managed_access:
			cuda::memory::managed::initial_visibility_t::to_all_devices );

   results = cuda::memory::managed::make_unique<float2[]>( results_size );
   expected_results = cuda::memory::managed::make_unique<float2[]>( results_size );

   /*auto current_device = cuda::device::current::get();*/
   /*vals = cuda::memory::device::make_unique<float2[]>(current_device, config.num_vals);*/
   /*results = cuda::memory::device::make_unique<float2[]>(current_device, config.num_results);*/
   
}


// Move Constructor
SlidingWindow::SlidingWindow( SlidingWindow&& other ) noexcept {
   num_vals = other.num_vals;
   window_size = other.window_size;
   kernel_sel = other.kernel_sel;
   debug = other.debug;
   num_results = other.num_results;

   vals = std::move( other.vals );
   results = std::move(other.results);
   expected_results = std::move(other.expected_results);
   
   vals = std::move( other.vals );
   results = std::move( other.results );

   other.num_vals = 0;
   other.window_size = 0;
   other.num_results = 0;
   other.kernel_sel = KernelSel::rolled_sel;

   other.vals.reset();
   other.results = nullptr;
   other.expected_results = nullptr;

   other.vals = nullptr;
   other.results = nullptr;
   
}


// Move Assignemt constructor
SlidingWindow& SlidingWindow::operator=( SlidingWindow&& other ) noexcept {
   if ( this != &other ) {
      num_vals = other.num_vals;
      window_size = other.window_size;
      kernel_sel = other.kernel_sel;
      num_results = other.num_results;
      debug = other.debug;

      vals = std::move( other.vals );
      results = std::move( other.results );
      expected_results = std::move( other.expected_results );

      vals = std::move( other.vals );
      results = std::move( other.results );
      
      other.vals.reset();
      other.results.reset();
      other.expected_results.reset();

      other.num_vals = 0;
      other.window_size = 0;
      other.num_results = 0;
      other.kernel_sel = KernelSel::rolled_sel;
   }
   return *this;
}

SlidingWindow::~SlidingWindow() {

   vals.reset();
   results.reset();
   expected_results.reset();

   vals.reset();
   results.reset();
   num_vals = 0;
   num_results = 0;
   kernel_sel = KernelSel::rolled_sel;
   window_size = 0;
}


void SlidingWindow::gen_vals() {
   for( int index = 0; index < num_vals; ++index ) {
      vals.get()[index].x = (float)index;
      vals.get()[index].y = (float)index;
   }   
}


void SlidingWindow::gen_expected() {
   
   expected_results.get()[0].x = 0.f;
   expected_results.get()[0].y = 0.f;

   for( int s_index = 0; s_index < window_size; ++s_index ) {
      expected_results.get()[0].x += vals.get()[s_index].x;
      expected_results.get()[0].y += vals.get()[s_index].y;
   }

   float prev_x_sum = expected_results.get()[0].x;
   float prev_y_sum = expected_results.get()[0].y;

   for ( int index = 1; index < num_results; ++index ) {
      expected_results.get()[index].x = prev_x_sum - vals.get()[index-1].x + vals.get()[index + window_size - 1].x;
      expected_results.get()[index].y = prev_y_sum - vals.get()[index-1].y + vals.get()[index + window_size - 1].y;
      prev_x_sum = expected_results.get()[index].x;
      prev_y_sum = expected_results.get()[index].y;
   }


   for ( int index = 0; index < num_results; ++index ) {
      expected_results.get()[index].x /= window_size;
      expected_results.get()[index].y /= window_size;
   }
   
   if ( debug ) {
      std::cout << __func__ << "(): expected_results.get()[0] = {" 
         << expected_results.get()[0].x
         << ", " 
         << expected_results.get()[0].y << "}\n";
   }
}


void SlidingWindow::check_results() {
   try {
      constexpr float comp_prec = 0.5;
      constexpr int num_places = 9; 
      for ( int index = 0; index < num_results; ++index ) {
      
         if (( fabs( expected_results.get()[index].x - results.get()[index].x ) > comp_prec) || 
            ( fabs( expected_results.get()[index].y - results.get()[index].y ) > comp_prec )) {
            
            std::cout << "Actual Result " 
               << index 
               << std::setprecision( num_places )
               << ": {" << results.get()[index].x
               << std::setprecision( num_places )
               << "," << results.get()[index].y 
               << "} does not match the expected "
               << index 
               << std::setprecision( num_places )
               << ": {" << expected_results.get()[index].x
               << std::setprecision( num_places )
               << "," << expected_results.get()[index].y
               << "}\n";

            throw std::runtime_error("Result too different from expected.");
         } 
      } // for ( int index = 0; index < num_results; ++index ) {
   } catch( std::exception& ex ) {
      std::cout << "ERROR: " << ex.what() << "\n"; 
   }
}


void SlidingWindow::run() {

   gen_vals();
   gen_expected();

   if ( debug ) {
      std::cout << "Num Vals = " << num_vals << "\n";
      std::cout << "Window Size = " << window_size << "\n";
      std::cout << "Num Vals Windowed = " << num_results << "\n";
      std::cout << "Kernel Select = " << get_kernel_sel_str( kernel_sel ) << "\n";
      std::cout << "Vals: \n";
      for( int index = 0; index < num_vals; ++index ) {
         std::cout << "[" << index << "] {" 
            << vals.get()[index].x << ", " << vals.get()[index].y << "}\n ";
      } 
      std::cout << "\n"; 
   }

   int threads_per_block = 1024;
   int blocks_per_grid = CEILING( num_results, threads_per_block );

   cuda::grid_dimensions_t grid_dims( blocks_per_grid, 1, 1 );
   cuda::grid_dimensions_t block_dims( threads_per_block, 1, 1 );
   cuda::memory::shared::size_t s_mem_size( 0u );
   cuda::launch_configuration_t launch_configuration = cuda::make_launch_config( grid_dims, 
         block_dims, s_mem_size );

   Time_Point start;
   Duration_ms duration_ms;
   
   //size_t size_vals = num_vals * sizeof(float2);
   //size_t size_results = num_results * sizeof(float2);

   start = Steady_Clock::now();
   
   if ( debug ) {
      std::cout << "CUDA kernel " << get_kernel_sel_str( kernel_sel ) 
         << " launch with " << blocks_per_grid
         << " blocks of " << threads_per_block << " threads\n";
   }
   
   auto device = cuda::device::current::get();
   auto stream = device.create_stream(cuda::stream::async);
   
   // TODO: Figure out how to make the switch only set the 
   // Kernel variable and then do cuda::launch() after the
   // switch block
   switch( kernel_sel ) {
      case KernelSel::rolled_sel:
         std::cout << "Rolled Kernel\n";
         stream.enqueue.kernel_launch(
            sliding_window,
            launch_configuration,
            results.get(), vals.get(), window_size, num_results
         );
         break;
      case KernelSel::unrolled2x_sel:
         std::cout << "Unrolled 2x Kernel\n";
         stream.enqueue.kernel_launch(
            sliding_window_unrolled_2x_inner,
            launch_configuration,
            results.get(), vals.get(), window_size, num_results
         );
         break;
      case KernelSel::unrolled4x_sel:
         std::cout << "Unrolled 4x Kernel\n";
         stream.enqueue.kernel_launch(
            sliding_window_unrolled_4x_inner,
            launch_configuration,
            results.get(), vals.get(), window_size, num_results
         );
         break;
      case KernelSel::unrolled8x_sel:
         std::cout << "Unrolled 8x Kernel\n";
         stream.enqueue.kernel_launch(
            sliding_window_unrolled_8x_inner,
            launch_configuration,
            results.get(), vals.get(), window_size, num_results
         );
         break;
      default:
         std::cout << "Rolled Kernel\n";
         stream.enqueue.kernel_launch(
            sliding_window,
            launch_configuration,
            results.get(), vals.get(), window_size, num_results
         );
         break;
   } // end of switch
   

   stream.enqueue.memory_attachment(results.get());
   stream.synchronize();

   duration_ms = Steady_Clock::now() - start;
   std::cout << num_vals << "," << window_size << "," << duration_ms.count() << ";\n";

   //check_results();

   if ( debug ) {
      std::cout << "Results: \n";
      for( int index = 0; index < num_results; ++index ) {
         std::cout << "[" << index << "]: {" << results.get()[index].x << ", " 
            << results.get()[index].y << "}\n";
      } 
      std::cout << "\n"; 
   }   

}
