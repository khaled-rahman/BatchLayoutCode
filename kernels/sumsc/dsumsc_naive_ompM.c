#include<omp.h>
#define NUMOFTHREADS 18

void ATL_USUMSC(long M, const double *X, long N, const double *Y, double *Z, long incX, long incY )
{
   long i, j;
   for (j=0; j < N; j++)
   {
      double y = Y[j]; 
      double sum=0.0;
      #pragma omp parallel for simd reduction(+:sum)
      for (i=0; i < M; i++)
         sum += y * X[i];
      Z[j] = sum;
   }
   
}
