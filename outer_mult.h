#include "CSC.h"
#include "CSR.h"
#include "utility.h"
#include <omp.h>
#include <algorithm>
#include <unistd.h>

uint64_t getTotalSystemMemory()
{
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return pages * page_size;
}


template <typename IT, typename NT>
uint64_t getFlop(const CSC<IT,NT> & A, const CSR<IT,NT> & B)
{
    uint64_t flop = 0;
#pragma omp parallel for reduction(+:flop)
    for (IT i=0; i < A.cols; ++i)
    {
        IT colnnz = A.colptr[i+1] - A.colptr[i];
        IT rownnz = B.rowptr[i+1] - B.rowptr[i];
        flop += (colnnz * rownnz);
    }
    return flop;
}

template <typename IT, typename NT>
void getFlopByRow(const CSC<IT,NT> & A, const CSR<IT,NT> & B, IT startIdx, IT endIdx, vector<IT> & flopsPerRow, uint64_t& totalFlop)
{

    for (IT i=startIdx; i < endIdx; ++i) // outer product of ith row of A and ith column of B
    {
        IT rownnz = B.rowptr[i+1] - B.rowptr[i];
        IT colnnz = A.colptr[i+1] - A.colptr[i];
        totalFlop += (colnnz * rownnz);
        for (IT j = A.colptr[i]; j < A.colptr[i + 1]; ++j) // For all the nonzeros of the ith column
        {
            IT row = A.rowids[j];
            flopsPerRow[row] += rownnz;
        }
    }
}


template <typename IT, typename NT>
int64_t getReqMemory(const CSC<IT,NT> & A, const CSR<IT,NT> & B)
{
    uint64_t flop = getFlop(A,B);

}



template <typename IT, typename NT>
void OuterSpGEMM_stage(const CSC<IT,NT> & A, const CSR<IT,NT> & B, IT startIdx, IT endIdx, vector<IT> flopSpace)
{
    
    vector<IT> counter(A.rows,0);
    double total = 1;
    int64_t count = 0;
    for (IT i=startIdx; i < endIdx; ++i) // outer product of ith row of A and ith column of B
    {
        for (IT j = A.colptr[i]; j < A.colptr[i + 1]; ++j) // For all the nonzeros of the ith column
        {
            IT row = A.rowids[j];
            for (IT k = B.rowptr[i]; k < B.rowptr[i + 1]; ++k) // For all the nonzeros of the ith row
            {
                IT col = B.colids[k];
                //NT val = multop(A.values[j], B.values[k]);
                NT val = A.values[j] * B.values[k];
                // store it somewhere
                // we know the storage requirement
                // no symbolic is needed
                flopSpace[count++] = col;
                //flopSpace[kk++] = val;
                total = total + col % 1000000;
            }
        }
    }
    
    cout << total << endl;
}


template <typename IT, typename NT>
void mtxstream(const CSC<IT,NT> & A, const CSR<IT,NT> & B)
{
    //int64_t N = A.nnz;
    IT* a = A.rowids;
    IT* b = B.colids;
    double start = omp_get_wtime();
    int niter = 10;
    for (int iter = 0; iter < niter; ++iter)
    {
        for (IT i=0; i < A.cols; ++i) // outer product of ith row of A and ith column of B
        {
            //IT rownnz = B.rowptr[i+1] - B.rowptr[i];
            //IT colnnz = A.colptr[i+1] - A.colptr[i];
            //totalFlop += (colnnz * rownnz);
            //totalFlop += (B.rowptr[i] - A.colptr[i]);
            IT start = A.colptr[i];
            IT end = A.colptr[i+1];
            for (IT j = start; j < end; ++j) // For all the nonzeros of the ith column
            {
                a[j] = b[j];
            }
        }
    }
    
    
    
    
    double end = omp_get_wtime();
    double msec = ((end - start) * 1000)/niter;
    double N = A.nnz + A.rows ;
    int itemsize = sizeof(IT);

    double bandwidth = 2 * (double)N * itemsize / 1024 / 1024 / msec;
    cout << "bandwidth : " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << endl;
}

template<typename NT>
void stream(vector<NT> a, vector<NT> b, int itemsize)
{
    int64_t N = a.size();
    double start = omp_get_wtime();
    int niter = 10;
    for (int iter = 0; iter < niter; ++iter)
    {
//#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            a[i] = b[i];
        }
    }
    double end = omp_get_wtime();
    double msec = ((end - start) * 1000)/niter;
    
    
    double bandwidth = 2 * (double)N * itemsize / 1024 / 1024 / msec;
    cout << "StreamTest : " << bandwidth << " [GB/sec], " << msec << " [milli seconds]" << endl;
}


/**
 ** Count flop of SpGEMM between A and B in CSC format
 **/
template <typename IT, typename NT>
int64_t OuterSpGEMM(const CSC<IT,NT> & A, const CSR<IT,NT> & B)
{
    //int64_t flop = getFlop(A,B);
    //getTotalSystemMemory();
    vector<IT> stream1(A.nnz,0);
    std::iota(stream1.begin(), stream1.end(), 0);
    vector<IT> stream2(A.nnz,0);
    std::iota(stream2.begin(), stream2.end(), 0);
    stream(stream1, stream2, sizeof(IT));
    
    uint64_t totalFlop = 0;
    vector<IT> flopsPerRow(A.rows, 0);
    getFlopByRow(A, B, 0, A.rows, flopsPerRow, totalFlop);
    
    /*
    vector<vector<IT>> flopSpace(A.rows);
    for(int i=0; i<A.rows; i++)
    {
        flopSpace[i].resize(flopsPerRow[i]);
    }*/
    
    
    vector<IT> flopSpace(totalFlop, 0);
    double start = omp_get_wtime();
    OuterSpGEMM_stage(A, B, 0, A.rows, flopSpace);
    double end = omp_get_wtime();
    double sec = end - start;
    double msec = sec * 1000;

    double mflop = (double)totalFlop / 1024 / 1024;
    double mflops = mflop / sec;
    double data = (double)totalFlop * 4 + A.nnz * 8 * 2 + A.nnz * 4 * 2;
    double bandwidth = 2 * data / 1024 / 1024 / msec;
    cout << "mflop: " << mflop << " mflops: " <<  mflops << endl;
    cout << "bandwidth: " << bandwidth << " [GB/sec] " << endl;
    
}
