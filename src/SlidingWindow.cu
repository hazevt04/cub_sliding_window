#include "SlidingWindow.cuh"


SlidingWindow::SlidingWindow(
      const int new_window_size,
      const std::vector<float2> new_vals
   ) {


   size_t vals_size = new_vals.size() * sizeof(float2);

   // Using CUDA Unified Memory
   // For data from CPU to GPU
   auto my_alloc_in = [](size_t my_size) { 
      void* ptr; 
      cudaMallocManaged((void**)&ptr, my_size, cudaMemAttachHost); 
      std::cout<<"\nmy_alloc_in\n"; 
      return ptr; 
   };
   // For data from GPU to CPU
   auto my_alloc_out = [](size_t my_size) { 
      void* ptr; 
      cudaMallocManaged((void**)&ptr, my_size); 
      std::cout<<"\nmy_alloc_out\n"; 
      return ptr; 
   };
   auto my_dealloc = [](float2* ptr) { 
      cudaFree(ptr); std::cout<<"\nmy_cuda_dealloc\n"; 
   };
   shared_p<decltype(my_dealloc), decltype(my_alloc_in)> vals((float2*)my_alloc_in(vals_size), my_dealloc); 
   shared_p<decltype(my_dealloc), decltype(my_alloc_out)> results((float2*)my_alloc_in(vals_size), my_dealloc); 

}
