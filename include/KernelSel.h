#ifndef __KERNELSEL_H__
#define __KERNELSEL_H__

#include "utils.h"

enum KernelSel: int {
   rolled_sel = 1,
   unrolled2x_sel = 2,
   unrolled4x_sel = 4,
   unrolled8x_sel = 8
};

inline std::string get_kernel_sel_str( KernelSel sel ) {
   std::string kernel_sel_str = "";
   switch( sel ) {
      case( KernelSel::rolled_sel ):
         kernel_sel_str = "Rolled Kernel";
         break;
      case( KernelSel::unrolled2x_sel ):
         kernel_sel_str = "Unrolled 2x Kernel";
         break;
      case( KernelSel::unrolled4x_sel ):
         kernel_sel_str = "Unrolled 4x Kernel";
         break;
      case( KernelSel::unrolled8x_sel ):
         kernel_sel_str = "Unrolled 8x Kernel";
         break;
      default:
         kernel_sel_str = "Unknown";
         break;
   }
   return kernel_sel_str;
}

#endif // end of #ifndef __KERNELSEL_H__
