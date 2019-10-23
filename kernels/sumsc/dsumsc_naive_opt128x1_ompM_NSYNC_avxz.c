#include<omp.h>
#include<immintrin.h>

#include<stdio.h>


#ifndef NT
   #define NT 18
#endif


void ATL_USUMSC(long M, const double *X, long N, const double *Y, double *Z, long incX, long incY )
{
   int m, n, k;
   long j;

   omp_set_num_threads(NT);

   for (j=0; j < N; j+=128)
   {
/*
 *    To avoid synchronization in reduction
 *    NOTE: we have 16 reduction AVX512 SIMD vector... 
 *          no need to pad to avoid false sharing because each vector is 
 *          64 bytes which is equal to the chache-line size 
 *    Storage used to store the sum: NT * 1KB 
 */
      double sum[NT][16][8]; /* 16 vector sum for each thread */

      register __m512d vsum; 
      register __m512d vy0;
      register __m512d vy1;
      register __m512d vy2;
      register __m512d vy3;
      register __m512d vy4;
      register __m512d vy5;
      register __m512d vy6;
      register __m512d vy7;
      register __m512d vy8;
      register __m512d vy9;
      register __m512d vy10;
      register __m512d vy11;
      register __m512d vy12;
      register __m512d vy13;
      register __m512d vy14;
      register __m512d vy15;
/*
 *    init the storage for sum, NT compile-time. so, compiler will unroll   
 */
      for (m = 0; m < NT; m++)
            for (n = 0; n < 16; n++)
               for (k = 0; k < 8; k++)
                  sum[m][n][k] = 0.0;


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
      
      #pragma omp parallel 
      { 
         long i; 
         int id = omp_get_thread_num(); 
         int nthds = omp_get_num_threads(); 
         size_t chunksize = M / nthds; 
        
         register __m512d tsum; /* temp sum */
         register __m512d tvsum0 = _mm512_set1_pd(0.0);
         register __m512d tvsum1 = _mm512_set1_pd(0.0);
         register __m512d tvsum2 = _mm512_set1_pd(0.0);
         register __m512d tvsum3 = _mm512_set1_pd(0.0);
         register __m512d tvsum4 = _mm512_set1_pd(0.0);
         register __m512d tvsum5 = _mm512_set1_pd(0.0);
         register __m512d tvsum6 = _mm512_set1_pd(0.0);
         register __m512d tvsum7 = _mm512_set1_pd(0.0);
         register __m512d tvsum8 = _mm512_set1_pd(0.0);
         register __m512d tvsum9 = _mm512_set1_pd(0.0);
         register __m512d tvsum10 = _mm512_set1_pd(0.0);
         register __m512d tvsum11 = _mm512_set1_pd(0.0);
         register __m512d tvsum12 = _mm512_set1_pd(0.0);
         register __m512d tvsum13 = _mm512_set1_pd(0.0);
         register __m512d tvsum14 = _mm512_set1_pd(0.0);
         register __m512d tvsum15 = _mm512_set1_pd(0.0);
/*
 *       NOTE: use chunk of consecutive data for each threads... 
 *       I'm using chunksize = M/nthds... meaning single block for each thread
 *       How would it effect the performance if each thread handles multiple 
 *       blocks with fixed smaller size (like, openmp static chunk )?? 
 *       NOTE: however, don't do this, poor cache utilization: 
                  for (i=id; i < M; i+=nthds) 
               meaning, it would use one element on each cacheline and throw all
               the data in that cacheline away for X  
 */
         
         for (i=id*chunksize; i < (id+1)*chunksize; i++)
         {
            double x;
            register __m512d vx0; 
            x = X[i];
            
            vx0 = _mm512_set1_pd(x);
         
            tvsum0 = _mm512_fmadd_pd(vx0, vy0, tvsum0);
            tvsum1 = _mm512_fmadd_pd(vx0, vy1, tvsum1);
            tvsum2 = _mm512_fmadd_pd(vx0, vy2, tvsum2);
            tvsum3 = _mm512_fmadd_pd(vx0, vy3, tvsum3);
            tvsum4 = _mm512_fmadd_pd(vx0, vy4, tvsum4);
            tvsum5 = _mm512_fmadd_pd(vx0, vy5, tvsum5);
            tvsum6 = _mm512_fmadd_pd(vx0, vy6, tvsum6);
            tvsum7 = _mm512_fmadd_pd(vx0, vy7, tvsum7);
            tvsum8 = _mm512_fmadd_pd(vx0, vy8, tvsum8);
            tvsum9 = _mm512_fmadd_pd(vx0, vy9, tvsum9);
            tvsum10 = _mm512_fmadd_pd(vx0, vy10, tvsum10);
            tvsum11 = _mm512_fmadd_pd(vx0, vy11, tvsum11);
            tvsum12 = _mm512_fmadd_pd(vx0, vy12, tvsum12);
            tvsum13 = _mm512_fmadd_pd(vx0, vy13, tvsum13);
            tvsum14 = _mm512_fmadd_pd(vx0, vy14, tvsum14);
            tvsum15 = _mm512_fmadd_pd(vx0, vy15, tvsum15);
         }
/*
 *       cleanup (M%NTHRDS) handled by single thread
 *       FIXME: make it parallel using logN threads each time 
 */
         #pragma omp single nowait 
         {
            for (i=nthds*chunksize; i < M; i++)
            {
               double x;
               register __m512d vx0; 
               
               x = X[i];
               vx0 = _mm512_set1_pd(x);
         
               tvsum0 = _mm512_fmadd_pd(vx0, vy0, tvsum0);
               tvsum1 = _mm512_fmadd_pd(vx0, vy1, tvsum1);
               tvsum2 = _mm512_fmadd_pd(vx0, vy2, tvsum2);
               tvsum3 = _mm512_fmadd_pd(vx0, vy3, tvsum3);
               tvsum4 = _mm512_fmadd_pd(vx0, vy4, tvsum4);
               tvsum5 = _mm512_fmadd_pd(vx0, vy5, tvsum5);
               tvsum6 = _mm512_fmadd_pd(vx0, vy6, tvsum6);
               tvsum7 = _mm512_fmadd_pd(vx0, vy7, tvsum7);
               tvsum8 = _mm512_fmadd_pd(vx0, vy8, tvsum8);
               tvsum9 = _mm512_fmadd_pd(vx0, vy9, tvsum9);
               tvsum10 = _mm512_fmadd_pd(vx0, vy10, tvsum10);
               tvsum11 = _mm512_fmadd_pd(vx0, vy11, tvsum11);
               tvsum12 = _mm512_fmadd_pd(vx0, vy12, tvsum12);
               tvsum13 = _mm512_fmadd_pd(vx0, vy13, tvsum13);
               tvsum14 = _mm512_fmadd_pd(vx0, vy14, tvsum14);
               tvsum15 = _mm512_fmadd_pd(vx0, vy15, tvsum15);
            }
         }
/*
 *       Reduction using the memory to avoid synchronization  
 *       NOTE: check assembly, compiler should get rid of the extra register
 *       and use memory variant of the ADD 
 *       NOTE: it's outside the loop... so, no need to optimize the pipeline 
 */
         /* sum them up */

         tsum = _mm512_loadu_pd(sum[id][0]);   
         tvsum0 = _mm512_add_pd(tvsum0, tsum); 
         _mm512_storeu_pd(sum[id], tvsum0);
         
         tsum = _mm512_loadu_pd(sum[id][1]);   
         tvsum1 = _mm512_add_pd(tvsum1, tsum); 
         _mm512_storeu_pd(sum[id][1], tvsum1);
         
         tsum = _mm512_loadu_pd(sum[id][2]);   
         tvsum2 = _mm512_add_pd(tvsum2, tsum); 
         _mm512_storeu_pd(sum[id][2], tvsum2);
         
         tsum = _mm512_loadu_pd(sum[id][3]);   
         tvsum3 = _mm512_add_pd(tvsum3, tsum); 
         _mm512_storeu_pd(sum[id][3], tvsum3);
         
         tsum = _mm512_loadu_pd(sum[id][4]);   
         tvsum4 = _mm512_add_pd(tvsum4, tsum); 
         _mm512_storeu_pd(sum[id][4], tvsum4);
         
         tsum = _mm512_loadu_pd(sum[id][5]);   
         tvsum5 = _mm512_add_pd(tvsum5, tsum); 
         _mm512_storeu_pd(sum[id][5], tvsum5);
              
         tsum = _mm512_loadu_pd(sum[id][6]);   
         tvsum6 = _mm512_add_pd(tvsum6, tsum); 
         _mm512_storeu_pd(sum[id][6], tvsum6);
         
         tsum = _mm512_loadu_pd(sum[id][7]);   
         tvsum7 = _mm512_add_pd(tvsum7, tsum); 
         _mm512_storeu_pd(sum[id][7], tvsum7);
         
         tsum = _mm512_loadu_pd(sum[id][8]);   
         tvsum8 = _mm512_add_pd(tvsum8, tsum); 
         _mm512_storeu_pd(sum[id][8], tvsum8);
         
         tsum = _mm512_loadu_pd(sum[id][9]);   
         tvsum9 = _mm512_add_pd(tvsum9, tsum); 
         _mm512_storeu_pd(sum[id][9], tvsum9);
         
         tsum = _mm512_loadu_pd(sum[id][10]);   
         tvsum10 = _mm512_add_pd(tvsum10, tsum); 
         _mm512_storeu_pd(sum[id][10], tvsum10);
         
         tsum = _mm512_loadu_pd(sum[id][11]);   
         tvsum11 = _mm512_add_pd(tvsum11, tsum); 
         _mm512_storeu_pd(sum[id][11], tvsum11);
         
         tsum = _mm512_loadu_pd(sum[id][12]);   
         tvsum12 = _mm512_add_pd(tvsum12, tsum); 
         _mm512_storeu_pd(sum[id][12], tvsum12);
         
         tsum = _mm512_loadu_pd(sum[id][13]);   
         tvsum13 = _mm512_add_pd(tvsum13, tsum); 
         _mm512_storeu_pd(sum[id][13], tvsum13);
         
         tsum = _mm512_loadu_pd(sum[id][14]);   
         tvsum14 = _mm512_add_pd(tvsum14, tsum); 
         _mm512_storeu_pd(sum[id][14], tvsum14);
         
         tsum = _mm512_loadu_pd(sum[id][15]);   
         tvsum15 = _mm512_add_pd(tvsum15, tsum); 
         _mm512_storeu_pd(sum[id][15], tvsum15);
      }
/*
 *    sum up all the work done by individual threads
 *    FIXME: make it parallel 
 */
      for (m=1; m < NT; m++)
         for (n=0; n < 16; n++)
            for (k=0; k < 8; k++)
               sum[0][n][k] += sum[m][n][k];


      
      vsum = _mm512_loadu_pd(sum[0][0]);   
      _mm512_storeu_pd(Z+j, vsum);
      
      vsum = _mm512_loadu_pd(sum[0][1]);   
      _mm512_storeu_pd(Z+j+8, vsum);
            
      vsum = _mm512_loadu_pd(sum[0][2]);   
      _mm512_storeu_pd(Z+j+2*8, vsum);

      vsum = _mm512_loadu_pd(sum[0][3]);   
      _mm512_storeu_pd(Z+j+3*8, vsum);

      vsum = _mm512_loadu_pd(sum[0][4]);   
      _mm512_storeu_pd(Z+j+4*8, vsum);

      vsum = _mm512_loadu_pd(sum[0][5]);   
      _mm512_storeu_pd(Z+j+5*8, vsum);
      
      vsum = _mm512_loadu_pd(sum[0][6]);   
      _mm512_storeu_pd(Z+j+6*8, vsum);

      vsum = _mm512_loadu_pd(sum[0][7]);   
      _mm512_storeu_pd(Z+j+7*8, vsum);
      
      vsum = _mm512_loadu_pd(sum[0][8]);   
      _mm512_storeu_pd(Z+j+8*8, vsum);
      
      vsum = _mm512_loadu_pd(sum[0][9]);   
      _mm512_storeu_pd(Z+j+9*8, vsum);
      
      vsum = _mm512_loadu_pd(sum[0][10]);   
      _mm512_storeu_pd(Z+j+10*8, vsum);
      
      vsum = _mm512_loadu_pd(sum[0][11]);   
      _mm512_storeu_pd(Z+j+11*8, vsum);
      
      vsum = _mm512_loadu_pd(sum[0][12]);   
      _mm512_storeu_pd(Z+j+12*8, vsum);

      vsum = _mm512_loadu_pd(sum[0][13]);   
      _mm512_storeu_pd(Z+j+13*8, vsum);
      
      vsum = _mm512_loadu_pd(sum[0][14]);   
      _mm512_storeu_pd(Z+j+14*8, vsum);
      
      vsum = _mm512_loadu_pd(sum[0][15]);   
      _mm512_storeu_pd(Z+j+15*8, vsum);
   }
   
}
