/*
 * NOTE: Majedul: The purpose of this header file is to hide all architecture 
 * dependent implementation of intrinsic vector codes 
 */
#ifndef _SIMD_H_
#define _SIMD_H_

#ifdef BLC_X86
   #if defined(BLC_AVXZ) || defined(BLC_AVX512)

   #elif defined(BLC_AVX2) || defined(BLC_AVXMAC) 

   #elif defined(BLC_AVX)  // no support for FMAC 

   #elif defined(BLC_SSE2)

   #elif defined(BLC_SSE1)

#elif defined(BLC_VSX)  // openPower vector unit  

#elif defined(BLC_ARM64) // arm64 machine 

#elif defined(BLC_FRCGNUVEC) // GNUVEC by GCC  

#else
   #error "Unsupported Architecture!"
#endif


#endif
