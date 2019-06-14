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
#include "../CSR.h"
#include "../multiply.h"

#ifdef KNL_EXE
#include "../hash_mult.h"
#elif defined HW_EXE
#include "../hash_mult_hw.h"
#else
#include "../hash_mult_hw.h"
#endif

#include "commonutility.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int

int main(int argc, char* argv[]){

	const bool sortOutput = true;
	vector<int> tnums;
	CSR<INDEXTYPE, VALUETYPE> A_csr;
	tnums = {1, 2, 4, 8, 16, 32, 64};
	SetInputMatricesAsCSR(A_csr, argv);
	A_csr.Sorted();
	
	return 0;
}
