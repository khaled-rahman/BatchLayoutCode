/*
 * NOTE: Majedul: The purpose of this header file is to hide all architecture 
 * dependent implementation of intrinsic vector codes 
 */
#ifndef _SIMD_H_
#define _SIMD_H_

#define BLC_X86   
#define BLC_AVXZ 
/*#define BLC_AVX2 */
 /*
  *   inst format: inst(dist, src1, src2)
  */
#ifdef BLC_X86
   #if defined(BLC_AVXZ) || defined(BLC_AVX512) /* avx512f */
      #include<immintrin.h>
      #if VALUETPYE == double
         #define VLEN 8
/*
 *       AVX512 double precision 
 */
         #define VTYPE __m512d 
         #define BCL_vldu(v_, p_) v_ = _mm512_loadu_pd(p_) 
         #define BCL_vld(v_, p_) v_ = _mm512_load_pd(p_) 
         #define BCL_vzero(v_) v_ = _mm512_setzero_pd() 
         #define BCL_vstu(p_, v_) _mm512_storeu_pd(p_, v_) 
         #define BCL_vst(p_, v_)  _mm512_store_pd(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm512_set1_pd(*(p_))
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm512_add_pd(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm512_sub_pd(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm512_mul_pd(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm512_div_pd(s1_, s2_)
         #define BCL_vmac(d_, s1_, s2_) d_ = _mm512_fmadd_pd(s1_, s2_, d_)
         #define BCL_vrcp(d_) d_ = _mm512_rcp14_pd(d_); // reciprocal 
         #define BCL_maskz_vrcp(d_, k_) d_ = _mm512_rcp14_pd(k_, d_); // reciprocal 
         #define BCL_imaskz_vrcp(d_, ik_) \
         {  __mmask8 k0_ = _cvtu32_mask8(ik_); \
            d_ = _mm512_maskz_rcp14_pd(k0_, d_);\
         }
         #define BCL_cvtint2mask(k_, ik) k_ = _cvtu32_mask8(ik_) 
/*
 *       VVRSUM codes from ATLAS 
 */
         /* vector to scalar */
         #define BCL_vrsum1(d_, s_) \
         { __m256d t0_, t1_; __m128d x0_, x1_; \
            t0_ = _mm512_extractf64x4_pd(s_, 0); \
            t1_ = _mm512_extractf64x4_pd(s_, 1); \
            t0_ = _mm256_add_pd(t0_, t1_); \
            x0_ = _mm256_extractf128_pd(t0_, 0); \
            x1_ = _mm256_extractf128_pd(t0_, 1); \
            x0_ = _mm_add_pd(x0_, x1_); \
            x0_ = _mm_hadd_pd(x0_, x0_); \
            d_ = x0_[0];  \
         }
      
      #elif VALUETYPE == float
         #define VLEN 16
         #define VTYPE __m512 
         #define BCL_cvtint2mask(ik_, k_) k_ = _cvtu32_mask8(ik_) 
         #define BCL_vldu(v_, p_) v_ = _mm512_loadu_ps(p_) 
         #define BCL_vld(v_, p_) v_ = _mm512_load_ps(p_) 
         #define BCL_vzero(v_) v_ = _mm512_setzero_ps() 
         #define BCL_vstu(p_, v_) _mm512_storeu_ps(p_, v_) 
         #define BCL_vst(p_, v_)  _mm512_store_ps(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm512_set1_ps(*(p_))
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm512_add_ps(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm512_sub_ps(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm512_mul_ps(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm512_div_ps(s1_, s2_)
         #define BCL_vmac(d_, s1_, s2_) d_ = _mm512_fmadd_ps(s1_, s2_, d_)
         #define BCL_vrcp(d_) d_ = _mm512_rcp14_ps(d_); /* reciprocal */
         #define BCL_maskz_vrcp(d_, k_) d_ = _mm512_maskz_rcp14_ps(k_, d_); /* reciprocal */
         #define BCL_imaskz_vrcp(d_, ik_) \
         {  __mmask8 k0_ = _cvtu32_mask8(ik_); \
            d_ = _mm512_maskz_rcp14_ps(k0_, d_);\
         }
      #else
         #error "Unsupported Value Type!"
      #endif
/*
 * AVX 
 */
   #elif defined(BLC_AVX2) || defined(BLC_AVXMAC) || defined(BLC_AVX) 
      #include<immintrin.h>
      #if defined(BLC_AVX2) || defined(BLC_AVXMAC)
         #define ArchHasMAC 
      #endif
      #if VALUETPYE == double
         #define VLEN 4
         #define VTYPE __m256d 
         #define BCL_vldu(v_, p_) v_ = _mm256_loadu_pd(p_) 
         #define BCL_vld(v_, p_) v_ = _mm256_load_pd(p_) 
         #define BCL_vzero(v_) v_ = _mm256_setzero_pd() 
         #define BCL_vstu(p_, v_) _mm256_storeu_pd(p_, v_) 
         #define BCL_vst(p_, v_)  _mm256_store_pd(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm256_set1_pd(*(p_))
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm256_add_pd(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm256_sub_pd(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm256_mul_pd(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm256_div_pd(s1_, s2_)
         #ifdef ArchHasMAC
            #define BCL_vmac(d_, s1_, s2_) d_ = _mm256_fmadd_pd(s1_, s2_, d_)
         #else
            #define BCL_vmac(d_, s1_, s2_) \
            {  VTYPE vt_; \
               vt_ = _mm256_mul_pd(s1_, s2_); \
               d_ = _mm256_add_pd(vt_, d_); \
            }
         #endif
         /* NOTE: no reciprocal for double precision, only for single precision 
         //#define BCL_vrcp(d_) d_ = _mm256_rcp14_pd(d_); // reciprocal */
         #define BCL_vrcp(d_) \
         {   VTYPE _vx = _mm256_set1_pd(1.0); \
             d_ = _mm256_div_pd(_vx, d_); \
         }
         /* #define BCL_maskz_vrcp(k_, d_) d_ = _mm256_rcp14_pd(k_, d_); // reciprocal
         // need to test */
         #define BCL_imaskz_vrcp(d_, ik_) \
         {  VTYPE v0_ = _mm256_setzero_pd();\
            VTYPE v1_ = _mm256_set1_pd(1.0);\
            d_ = _mm256_blend_pd(d_, v1_, ik_); \
            d_ = _mm256_div_pd(v1_, d_); \
            d_ = _mm256_blend_pd(d_, v0_, ik_); \
         }
         /*#define BCL_cvtint2mask(k_, ik) k_ = _cvtu32_mask8(ik_) */
      #elif VALUETYPE == float
         #define VLEN 8
         #define VTYPE __m256 
         #define BCL_vldu(v_, p_) v_ = _mm256_loadu_ps(p_) 
         #define BCL_vld(v_, p_) v_ = _mm256_load_ps(p_) 
         #define BCL_vzero(v_) v_ = _mm256_setzero_ps() 
         #define BCL_vstu(p_, v_) _mm256_storeu_ps(p_, v_) 
         #define BCL_vst(p_, v_)  _mm256_store_ps(p_, v_) 
         #define BCL_vbcast(v_, p_) v_ = _mm256_set1_ps(*(p_))
         #define BCL_vadd(d_, s1_, s2_) d_ = _mm256_add_ps(s1_, s2_)
         #define BCL_vsub(d_, s1_, s2_) d_ = _mm256_sub_ps(s1_, s2_)
         #define BCL_vmul(d_, s1_, s2_) d_ = _mm256_mul_ps(s1_, s2_)
         #define BCL_vdiv(d_, s1_, s2_) d_ = _mm256_div_ps(s1_, s2_)
         #ifdef ArchHasMAC
            #define BCL_vmac(d_, s1_, s2_) d_ = _mm256_fmadd_ps(s1_, s2_, d_)
         #else
            #define BCL_vmac(d_, s1_, s2_) \
            {  VTYPE vt_; \
               vt_ = _mm256_mul_ps(s1_, s2_); \
               d_ = _mm256_add_ps(vt_, d_); \
            }
         #endif
         /* NOTE: no reciprocal for double precision, only for single precision */
         #if 1
            #define BCL_vrcp(d_) d_ = _mm256_rcp_ps(d_); /* reciprocal */
         #else
            #define BCL_vrcp(d_) \
            {  VTYPE _vx = _m256_set1_ps(1.0); \
               d_ = _m256_div_ps(_vx, d_); \
            }
         #endif
         /* #define BCL_maskz_vrcp(k_, d_) d_ = _mm256_rcp14_ps(k_, d_); // reciprocal
         // need to test */
         #if 1
            #define BCL_imaskz_vrcp(d_, ik_) \
            {  VTYPE v0_ = _mm256_setzero_ps();\
               VTYPE v1_ = _mm256_setzero+ps(1.0);\
               d_ = _mm256_blend_ps(d_, v1_, ik_); \
               d_ = _mm256_rcp_ps(d_); \
               d_ = _mm256_blend_ps(d_, v0_, ik_); \
            }
         #else
            #define BCL_imaskz_vrcp(d_, ik_) \
            {  VTYPE v0_ = _mm256_setzero_ps();\
               VTYPE v1_ = _mm256_setzero+ps(1.0);\
               d_ = _mm256_blend_ps(d_, v1_, ik_); \
               d_ = _mm256_div_ps(v1_, d_); \
               d_ = _mm256_blend_ps(d_, v0_, ik_); \
            }
         #endif
      #else
         #error "Unsupported Value Type!"
      #endif
   #elif defined(BLC_SSE2) || defined(BLC_SSE3)
      #if VALUETPYE == double
         #define VLEN 2
      #elif VALUETYPE == float
         #define VLEN 4
      #else
         #error "Unsupported Value Type!"
      #endif
   #elif defined(BLC_SSE1)
      #if VALUETYPE == float
         #define VLEN 4
      #else /* double not supported */
         #error "Unsupported Value Type!"
      #endif
   #elif defined(BLC_SSE1)
   #else
      #error "Unsupported X86 SIMD!"
   #endif

#elif defined(BLC_VSX)  /* openPower vector unit */
   #include <altivec.h>   
   #if VALUETPYE == double
      #define VLEN 2
      #define VTYPE vector double  
   #elif VALUETYPE == float
      #define VLEN 4
      #define VTYPE vector float 
   #else
      #error "Unsupported type for VSX SIMD!"
   #endif
   #define BLC_vldu(v_, p_) v_ = vec_vsx_ld(0, (VTYPE*)(p_)) 
   #define BLC_vld(v_, p_) v_ = vec_ld(0, (VTYPE*)(p_))  
   #define BLC_vzero(v_) v_ = vec_splats((VALUETYPE)0.0)
   #define BLC_vstu(p_, v_) vec_vsx_st(v_, 0, (VTYPE*)(p_))
   #define BLC_vst(p_, v_)  vec_st(v_, 0, (VTYPE*)(p_))
   #define BLC_vbcast(v_, p_) v_ =  vec_splats(*((VALUETYPE*)(p_)))
   #define BLC_vadd(d_, s1_, s2_) d_ =  vec_add(s1_, s2_) 
   #define BLC_vsub(d_, s1_, s2_) d_ =  vec_sub(s1_, s2_) 
   #define BLC_vmul(d_, s1_, s2_) d_ =  vec_mul(s1_, s2_) 
   #define BLC_vdiv(d_, s1_, s2_) d_ =  vec_div(s1_, s2_) 
   #define BLC_vmac(d_, s1_, s2_) d_ =  vec_madd(s1_, s2_, d_) 
   #define BCL_vrcp(d_) d_ = vec_re(d_); /* reciprocal */
   /* NOTE: need to use vec_se  
   //#define BCL_imaskz_vrcp(d_, ik_) \ */


#elif defined(BLC_ARM64) /* arm64 machine */

#elif defined(BLC_FRCGNUVEC) /* GNUVEC by GCC  */

#else
   #error "Unsupported Architecture!"
#endif


#endif
