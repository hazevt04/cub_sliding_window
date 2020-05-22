// C++ File for main

#include "sliding_window.cuh"


int main(int argc, char **argv) {
   int window_size = 3;
   int num_vals = 5;
   bool debug = true;
   int status = SUCCESS;

   try_func( status, "my_test failed", my_test( window_size, num_vals, debug ) );
}
