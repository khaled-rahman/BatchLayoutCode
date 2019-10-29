#include<immintrin.h>

void ATL_USUMSC(long M, const double *X, long N, const double *Y, double *Z, long incX, long incY )
/*
 * NO N CLEANUP: N must be multiple of 64 
 */
{
   long i, j, MM, NN;
   MM = (M/16)*16;
   NN = (N/64)*64;
/*
 * peeling one iteration of M loop to init Z[j] with x*Y[j] 
 */
   
   double x;
   register __m512d vx0;
   

   x = X[0];
   vx0 = _mm512_set1_pd(x);

   for (j=0; j < N; j+=8)
   {
      register __m512d vy0, vz0; 
      vy0 = _mm512_loadu_pd(Y+j);
      vz0 = _mm512_mul_pd(vx0, vy0);
       _mm512_storeu_pd(Z+j, vz0);
   }
 /*
  *   M-unroll = 16... so, peeling 15 iteration of M as cleanup 
  *   NOTE: cleanup doesn't matter much, still writing it using intrinsic 
  *   NOTE: this loop will bring all Y and Z in private cache 
  */
   for (i=1; i < 16; i++)
   {
      x = X[i]; 
      vx0 = _mm512_set1_pd(x);
      for (j=0; j < N; j+=8)
      {
         /*Z[j] += x * Y[j]; */
         register __m512d vy0, vz0; 
         
         vy0 = _mm512_loadu_pd(Y+j);
         vz0 = _mm512_loadu_pd(Z+j);
         /* all the fma are dependent.. need to unroll more.. but it's cleanup */
         vz0 = _mm512_fmadd_pd(vx0, vy0, vz0);
       _mm512_storeu_pd(Z+j, vz0);
      }
   }
/*
 * main loop : 16x64 unrolling  
 */
   for (i=16; i < MM; i+=16)
   {
/*
 *    set1_pd ... performs the broadcast 
 */
      double x; 
      register __m512d vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7;
      register __m512d vx8, vx9, vx10, vx11, vx12, vx13, vx14, vx15;
      
      x = X[i];
      vx0 = _mm512_set1_pd(x);
      x = X[i+1];
      vx1 = _mm512_set1_pd(x);
      x = X[i+2];
      vx2 = _mm512_set1_pd(x);
      x = X[i+3];
      vx3 = _mm512_set1_pd(x);
      x = X[i+4];
      vx4 = _mm512_set1_pd(x);
      x = X[i+5];
      vx5 = _mm512_set1_pd(x);
      x = X[i+6];
      vx6 = _mm512_set1_pd(x);
      x = X[i+7];
      vx7 = _mm512_set1_pd(x);
      
      x = X[i+8];
      vx8 = _mm512_set1_pd(x);
      x = X[i+9];
      vx9 = _mm512_set1_pd(x);
      x = X[i+10];
      vx10 = _mm512_set1_pd(x);
      x = X[i+11];
      vx11 = _mm512_set1_pd(x);
      x = X[i+12];
      vx12 = _mm512_set1_pd(x);
      x = X[i+13];
      vx13 = _mm512_set1_pd(x);
      x = X[i+14];
      vx14 = _mm512_set1_pd(x);
      x = X[i+15];
      vx15 = _mm512_set1_pd(x);
/*
 *    UN = 64 means, 8*VLEN unrolling... we need > 4VLEN unrolling to avoid
 *    stall in pipeline for FMAC
 */
      for (j=0; j < NN; j+=64)
      {
         __m512d vy0, vz0; 
         __m512d vy1, vz1; 
         __m512d vy2, vz2; 
         __m512d vy3, vz3;
         __m512d vy4, vz4; 
         __m512d vy5, vz5; 
         __m512d vy6, vz6; 
         __m512d vy7, vz7; 
         
         vy0 = _mm512_loadu_pd(Y+j);
         vy1 = _mm512_loadu_pd(Y+j+8);
         vy2 = _mm512_loadu_pd(Y+j+2*8);
         vy3 = _mm512_loadu_pd(Y+j+3*8);
         vy4 = _mm512_loadu_pd(Y+j+4*8);
         vy5 = _mm512_loadu_pd(Y+j+5*8);
         vy6 = _mm512_loadu_pd(Y+j+6*8);
         vy7 = _mm512_loadu_pd(Y+j+7*8);
         
         vz0 = _mm512_loadu_pd(Z+j);
         vz1 = _mm512_loadu_pd(Z+j+8);
         vz2 = _mm512_loadu_pd(Z+j+2*8);
         vz3 = _mm512_loadu_pd(Z+j+3*8);
         vz4 = _mm512_loadu_pd(Z+j+4*8);
         vz5 = _mm512_loadu_pd(Z+j+5*8);
         vz6 = _mm512_loadu_pd(Z+j+6*8);
         vz7 = _mm512_loadu_pd(Z+j+7*8);
         
         vz0 = _mm512_fmadd_pd(vx0, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx0, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx0, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx0, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx0, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx0, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx0, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx0, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx1, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx1, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx1, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx1, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx1, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx1, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx1, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx1, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx2, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx2, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx2, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx2, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx2, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx2, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx2, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx2, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx3, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx3, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx3, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx3, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx3, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx3, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx3, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx3, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx4, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx4, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx4, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx4, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx4, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx4, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx4, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx4, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx5, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx5, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx5, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx5, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx5, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx5, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx5, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx5, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx6, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx6, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx6, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx6, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx6, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx6, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx6, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx6, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx7, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx7, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx7, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx7, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx7, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx7, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx7, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx7, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx8, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx8, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx8, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx8, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx8, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx8, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx8, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx8, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx9, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx9, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx9, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx9, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx9, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx9, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx9, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx9, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx10, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx10, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx10, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx10, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx10, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx10, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx10, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx10, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx11, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx11, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx11, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx11, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx11, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx11, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx11, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx11, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx12, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx12, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx12, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx12, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx12, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx12, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx12, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx12, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx13, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx13, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx13, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx13, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx13, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx13, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx13, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx13, vy7, vz7);
         
         vz0 = _mm512_fmadd_pd(vx14, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx14, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx14, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx14, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx14, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx14, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx14, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx14, vy7, vz7);

         vz0 = _mm512_fmadd_pd(vx15, vy0, vz0);
         vz1 = _mm512_fmadd_pd(vx15, vy1, vz1);
         vz2 = _mm512_fmadd_pd(vx15, vy2, vz2);
         vz3 = _mm512_fmadd_pd(vx15, vy3, vz3);
         vz4 = _mm512_fmadd_pd(vx15, vy4, vz4);
         vz5 = _mm512_fmadd_pd(vx15, vy5, vz5);
         vz6 = _mm512_fmadd_pd(vx15, vy6, vz6);
         vz7 = _mm512_fmadd_pd(vx15, vy7, vz7);

         _mm512_storeu_pd(Z+j, vz0);
         _mm512_storeu_pd(Z+j+8, vz1);
         _mm512_storeu_pd(Z+j+2*8, vz2);
         _mm512_storeu_pd(Z+j+3*8, vz3);
         _mm512_storeu_pd(Z+j+4*8, vz4);
         _mm512_storeu_pd(Z+j+5*8, vz5);
         _mm512_storeu_pd(Z+j+6*8, vz6);
         _mm512_storeu_pd(Z+j+7*8, vz7);
      }
/*
 *    N loop cleanup .... not added assuming N is multiple of 64 for now 
 */
   }
/*
 * M loop cleanup 
 */
   for (; i < M; i++)
   {
      x = X[i]; 
      vx0 = _mm512_set1_pd(x);
      for (j=0; j < N; j+=8)
      {
         register __m512d vy0, vz0; 
         
         vy0 = _mm512_loadu_pd(Y+j);
         vz0 = _mm512_loadu_pd(Z+j);
         /* all the fma are dependent.. need to unroll more.. but it's cleanup */
         vz0 = _mm512_fmadd_pd(vx0, vy0, vz0);
       _mm512_storeu_pd(Z+j, vz0);
      }
   }
}
