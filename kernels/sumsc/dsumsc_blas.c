
#include <atlas_misc.h>
#include <atlas_level1.h>



void ATL_USUMSC(long M, const double *X, long N, const double *Y, double *Z, long incX, long incY )
{
   TYPE sum = Mjoin(PATL, asum)(M, X, incX); 
   Mjoin(PATL, copy)(N, Y, incY, Z, incY); 
   Mjoin(PATL, scal)(N, sum, Z, incY); 
}
