#include<omp.h>
#define NUMOFTHREADS 18

void ATL_USUMSC(long M, const double *X, long N, const double *Y, double *Z, long incX, long incY )
{
   long i, j;
   for (j=0; j < N; j+=16)
   {
      double y0, y1, y2, y3, y4, y5, y6, y7;
      double y8, y9, y10, y11, y12, y13, y14, y15;
      double sum0=0.0;
      double sum1=0.0;
      double sum2=0.0;
      double sum3=0.0;
      double sum4=0.0;
      double sum5=0.0;
      double sum6=0.0;
      double sum7=0.0;
      double sum8=0.0;
      double sum9=0.0;
      double sum10=0.0;
      double sum11=0.0;
      double sum12=0.0;
      double sum13=0.0;
      double sum14=0.0;
      double sum15=0.0;

      y0 = Y[j]; 
      y1 = Y[j+1]; 
      y2 = Y[j+2]; 
      y3 = Y[j+3]; 
      y4 = Y[j+4]; 
      y5 = Y[j+5]; 
      y6 = Y[j+6]; 
      y7 = Y[j+7]; 
      
      y8 = Y[j+8]; 
      y9 = Y[j+8+1]; 
      y10 = Y[j+8+2]; 
      y11 = Y[j+8+3]; 
      y12 = Y[j+8+4]; 
      y13 = Y[j+8+5]; 
      y14 = Y[j+8+6]; 
      y15 = Y[j+8+7]; 
      
      #pragma omp parallel for simd reduction(+: sum0) \
                                    reduction(+: sum1) \
                                    reduction(+: sum2) \
                                    reduction(+: sum3) \
                                    reduction(+: sum4) \
                                    reduction(+: sum5) \
                                    reduction(+: sum6) \
                                    reduction(+: sum7) \
                                    reduction(+: sum8) \
                                    reduction(+: sum9) \
                                    reduction(+: sum10) \
                                    reduction(+: sum11) \
                                    reduction(+: sum12) \
                                    reduction(+: sum13) \
                                    reduction(+: sum14) \
                                    reduction(+: sum15) 
      for (i=0; i < M; i++)
      {
         double x = X[i];
         sum0 += y0 * x;
         sum1 += y1 * x;
         sum2 += y2 * x;
         sum3 += y3 * x;
         sum4 += y4 * x;
         sum5 += y5 * x;
         sum6 += y6 * x;
         sum7 += y7 * x;
         sum8 += y8 * x;
         sum9 += y9 * x;
         sum10 += y10 * x;
         sum11 += y11 * x;
         sum12 += y12 * x;
         sum13 += y13 * x;
         sum14 += y14 * x;
         sum15 += y15 * x;
      }
      Z[j] = sum0;
      Z[j+1] = sum1;
      Z[j+2] = sum2;
      Z[j+3] = sum3;
      Z[j+4] = sum4;
      Z[j+5] = sum5;
      Z[j+6] = sum6;
      Z[j+7] = sum7;
      Z[j+8] = sum8;
      Z[j+8+1] = sum9;
      Z[j+8+2] = sum10;
      Z[j+8+3] = sum11;
      Z[j+8+4] = sum12;
      Z[j+8+5] = sum13;
      Z[j+8+6] = sum14;
      Z[j+8+7] = sum15;
   }
   
}
