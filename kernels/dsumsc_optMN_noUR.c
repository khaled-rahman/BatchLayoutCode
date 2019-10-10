#include<immintrin.h>

void ATL_USUMSC(long M, const double *X, long N, const double *Y, double *Z, long incX, long incY )
{
   long i, j;

   double x = X[0];
   for (j=0; j < N; j++)
      Z[j] = x * Y[j]; 
   
   for (i=1; i < M; i++)
   {
      double x = X[i];
      for (j=0; j < N; j++)
         Z[j] += x * Y[j]; 
   }
}
