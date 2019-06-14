#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <random>

//#include "overridenew.h"
#include "../utility.h"
#include "../CSC.h"
#include "../CSR.h"
#include "../IO.h"

using namespace std;

extern "C" {
#include "../GTgraph/R-MAT/defs.h"
#include "../GTgraph/R-MAT/init.h"
#include "../GTgraph/R-MAT/graph.h"
}

#ifndef _SAMPLE_COMMON_HPP_
#define _SAMPLE_COMMON_HPP_

#ifdef CPP
#define MALLOC "new"
#elif defined IMM
#define MALLOC "mm"
#elif defined TBB
#define MALLOC "tbb"
#else
#define MALLOC "tbb"
#endif

#define ITERS 10

template <class INDEXTYPE, class VALUETYPE>
void SetInputMatricesAsCSC(CSC<INDEXTYPE, VALUETYPE> &A_csc, char **argv)
{
	string inputname1;

    	A_csc.make_empty();
    
        inputname1 = argv[1];
        cout << "reading input matrices in text (ascii)... " << endl;
	cout << "Input File Directory:" << inputname1 << endl;
        ReadASCII( inputname1, A_csc );
        stringstream ss1(inputname1);
        string cur;
        
	vector<string> v1;
        while (getline(ss1, cur, '.')) {
            v1.push_back(cur);
        }
	inputname1 = v1[v1.size() - 1];
}

template <class INDEXTYPE, class VALUETYPE>
void SetInputMatricesAsCSR(CSR<INDEXTYPE, VALUETYPE> &A_csr, char **argv)
{
    CSC<INDEXTYPE, VALUETYPE> A_csc;

    A_csr.make_empty();

    SetInputMatricesAsCSC(A_csc, argv);

    A_csr = *(new CSR<INDEXTYPE, VALUETYPE>(A_csc));
}

template <class INDEXTYPE, class VALUETYPE>
void print(CSR<INDEXTYPE, VALUETYPE> &A_csr){
	cout << "Size of Row PTRS:" << A_csr.rows << ", Size of Col IDS:" << A_csr.nnz << endl;
	cout << "Rowptrs:";
        for(INDEXTYPE i = 0; i < A_csr.rows; i++){
                cout << A_csr.rowptr[i] << " ";
        }
        cout << endl;
        cout << "Colids:";
        for(INDEXTYPE i = 0; i < A_csr.nnz; i++){
                cout << A_csr.colids[i] << " ";
        }
        cout << endl;
        cout << "Values:";
        for(INDEXTYPE i = 0; i < A_csr.nnz; i++){
                cout << A_csr.values[i] << " ";
        }
        cout << endl;
}

#endif
