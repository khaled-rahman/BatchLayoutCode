#include<omp.h>
#define NUMOFTHREADS 18

void ATL_USUMSC(long M, const double *X, long N, const double *Y, double *Z, long incX, long incY )
{
   long i, j;
   #pragma omp parallel for simd schedule(static) 
   for (j=0; j < N; j++)
   {
      double y = Y[j]; 
      double sum=0.0;
      for (i=0; i < M; i++)
         sum += y * X[i];
      Z[j] = sum;
   }
   
}
