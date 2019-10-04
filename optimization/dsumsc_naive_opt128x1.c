#include<immintrin.h>

void ATL_USUMSC(long M, const double *X, long N, const double *Y, double *Z, long incX, long incY )
/*
 * NO CLEANUP, N must be multiple of 128 
 */
{
   long i, j;
   for (j=0; j < N; j+=128)
   {
      register __m512d vy0, vsum0;
      register __m512d vy1, vsum1;
      register __m512d vy2, vsum2;
      register __m512d vy3, vsum3;
      register __m512d vy4, vsum4;
      register __m512d vy5, vsum5;
      register __m512d vy6, vsum6;
      register __m512d vy7, vsum7;
      
      register __m512d vy8, vsum8;
      register __m512d vy9, vsum9;
      register __m512d vy10, vsum10;
      register __m512d vy11, vsum11;
      register __m512d vy12, vsum12;
      register __m512d vy13, vsum13;
      register __m512d vy14, vsum14;
      register __m512d vy15, vsum15;
      
      vy0 = _mm512_loadu_pd(Y+j);
      vy1 = _mm512_loadu_pd(Y+j+8);
      vy2 = _mm512_loadu_pd(Y+j+2*8);
      vy3 = _mm512_loadu_pd(Y+j+3*8);
      vy4 = _mm512_loadu_pd(Y+j+4*8);
      vy5 = _mm512_loadu_pd(Y+j+5*8);
      vy6 = _mm512_loadu_pd(Y+j+6*8);
      vy7 = _mm512_loadu_pd(Y+j+7*8);
       
      vy8 = _mm512_loadu_pd(Y+j+8*8);
      vy9 = _mm512_loadu_pd(Y+j+9*8);
      vy10 = _mm512_loadu_pd(Y+j+10*8);
      vy11 = _mm512_loadu_pd(Y+j+11*8);
      vy12 = _mm512_loadu_pd(Y+j+12*8);
      vy13 = _mm512_loadu_pd(Y+j+13*8);
      vy14 = _mm512_loadu_pd(Y+j+14*8);
      vy15 = _mm512_loadu_pd(Y+j+15*8);
      /*vsum = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};*/
      vsum0 = _mm512_set1_pd(0.0);
      vsum1 = _mm512_set1_pd(0.0);
      vsum2 = _mm512_set1_pd(0.0);
      vsum3 = _mm512_set1_pd(0.0);
      vsum4 = _mm512_set1_pd(0.0);
      vsum5 = _mm512_set1_pd(0.0);
      vsum6 = _mm512_set1_pd(0.0);
      vsum7 = _mm512_set1_pd(0.0);
      
      vsum8 = _mm512_set1_pd(0.0);
      vsum9 = _mm512_set1_pd(0.0);
      vsum10 = _mm512_set1_pd(0.0);
      vsum11 = _mm512_set1_pd(0.0);
      vsum12 = _mm512_set1_pd(0.0);
      vsum13 = _mm512_set1_pd(0.0);
      vsum14 = _mm512_set1_pd(0.0);
      vsum15 = _mm512_set1_pd(0.0);
/*
 *    want to multiple X traverse... so un N as much as possible
 */
      for (i=0; i < M; i++)
      {
         double x;
         register __m512d vx0; 
         x = X[i];
         vx0 = _mm512_set1_pd(x);
         
         vsum0 = _mm512_fmadd_pd(vx0, vy0, vsum0);
         vsum1 = _mm512_fmadd_pd(vx0, vy1, vsum1);
         vsum2 = _mm512_fmadd_pd(vx0, vy2, vsum2);
         vsum3 = _mm512_fmadd_pd(vx0, vy3, vsum3);
         vsum4 = _mm512_fmadd_pd(vx0, vy4, vsum4);
         vsum5 = _mm512_fmadd_pd(vx0, vy5, vsum5);
         vsum6 = _mm512_fmadd_pd(vx0, vy6, vsum6);
         vsum7 = _mm512_fmadd_pd(vx0, vy7, vsum7);
         vsum8 = _mm512_fmadd_pd(vx0, vy8, vsum8);
         vsum9 = _mm512_fmadd_pd(vx0, vy9, vsum9);
         vsum10 = _mm512_fmadd_pd(vx0, vy10, vsum10);
         vsum11 = _mm512_fmadd_pd(vx0, vy11, vsum11);
         vsum12 = _mm512_fmadd_pd(vx0, vy12, vsum12);
         vsum13 = _mm512_fmadd_pd(vx0, vy13, vsum13);
         vsum14 = _mm512_fmadd_pd(vx0, vy14, vsum14);
         vsum15 = _mm512_fmadd_pd(vx0, vy15, vsum15);
      }
      
      _mm512_storeu_pd(Z+j, vsum0);
      _mm512_storeu_pd(Z+j+8, vsum1);
      _mm512_storeu_pd(Z+j+2*8, vsum2);
      _mm512_storeu_pd(Z+j+3*8, vsum3);
      _mm512_storeu_pd(Z+j+4*8, vsum4);
      _mm512_storeu_pd(Z+j+5*8, vsum5);
      _mm512_storeu_pd(Z+j+6*8, vsum6);
      _mm512_storeu_pd(Z+j+7*8, vsum7);
      
      _mm512_storeu_pd(Z+j+8*8, vsum8);
      _mm512_storeu_pd(Z+j+9*8, vsum9);
      _mm512_storeu_pd(Z+j+10*8, vsum10);
      _mm512_storeu_pd(Z+j+11*8, vsum11);
      _mm512_storeu_pd(Z+j+12*8, vsum12);
      _mm512_storeu_pd(Z+j+13*8, vsum13);
      _mm512_storeu_pd(Z+j+14*8, vsum14);
      _mm512_storeu_pd(Z+j+15*8, vsum15);
   }
   
}
