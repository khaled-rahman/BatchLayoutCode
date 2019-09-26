#ifndef _ALGORITHMS_H_
#define _ALGORITHMS_H_

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
#include <map>
#include <unordered_map>
#include <cmath>
#include <vector>
#include <stack>
#include <string>
#include <sstream>
#include <random>
#include <limits>
#include <cassert>
#include <parallel/algorithm>
#include <parallel/numeric>
//#include <parallel/random_shuffle>

#include "../CSR.h"

#include "../utility.h"

#include "../sample/commonutility.h"
#include "../sample/Coordinate.h"

#include "../sample/MortonCode.h"
#include "../sample/BarnesHut.h"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int
#define CACHEBLOCK 4
#define MAXMIN 3.0
#define t 0.99
#define PI 3.14159265358979323846
static int PROGRESS = 0;

#pragma omp declare reduction(plus:Coordinate<VALUETYPE>:omp_out += omp_in) initializer(omp_priv = Coordinate<VALUETYPE>(0.0, 0.0))

class algorithms{
	public:
		CSR<INDEXTYPE, VALUETYPE> graph;
		Coordinate<VALUETYPE>  *nCoordinates, *prevCoordinates;
		VALUETYPE K = 1.0, C = 1.0, Shift=1.0, init;
		VALUETYPE minX, minY, maxX, maxY, W, threshold;
		string filename;
		string outputdir;
		string initfile;
	public:
	algorithms(CSR<INDEXTYPE, VALUETYPE> &A_csr, string input, string outputd, int init, double weight, double th, string ifile);
	void randInit();
	void initDFS();
	void fileInitialization();
	Coordinate<VALUETYPE> calcAttraction(INDEXTYPE i, INDEXTYPE j);
	Coordinate<VALUETYPE> calcRepulsion(INDEXTYPE i, INDEXTYPE n);
	VALUETYPE updateStepLength(VALUETYPE STEP, VALUETYPE ENERGY, VALUETYPE ENERGY0);	
	vector<VALUETYPE> seqForceDirectedAlgorithm(INDEXTYPE ITERATIONS);
	vector<VALUETYPE> seqAdjForceDirectedAlgorithm(INDEXTYPE ITERATIONS);
	vector<VALUETYPE> naiveParallelForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS);
	vector<VALUETYPE> miniBatchForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);
	vector<VALUETYPE> cacheBlockingminiBatchForceDirectedAlgorithmSD(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, int flag);
	vector<VALUETYPE> cacheBlockingminiBatchForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, int flag);
	vector<VALUETYPE> cacheBlockingminiBatchForceDirectedAlgorithmConverged(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, int flag);
	vector<VALUETYPE> LinLogcacheBlockingminiBatchForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);
	vector<VALUETYPE> FAcacheBlockingminiBatchForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);
	vector<VALUETYPE> BarnesHutApproximation(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, VALUETYPE TH, int flag);
	vector<VALUETYPE> approxForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);
	vector<VALUETYPE> approxCacheBlockBH(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);
	void print();
	void writeRepulsiveForce(vector<Coordinate<VALUETYPE> > &repulse, string f);
	void writeToFileBH(Coordinate<VALUETYPE> *tCoordinates, string f);
	void writeToFile(string f);
};
#endif
