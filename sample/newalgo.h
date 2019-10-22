#ifndef _NEWALGO_H_
#define _NEWALGO_H_

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


class newalgo{
	public:
		CSR<INDEXTYPE, VALUETYPE> graph;
		Coordinate<VALUETYPE>  *nCoordinates, *prevCoordinates;
		VALUETYPE *blasX, *blasY;
		VALUETYPE K = 1.0, C = 1.0, Shift=1.0, init;
		VALUETYPE minX, minY, maxX, maxY, W, threshold;
		string filename;
		string outputdir;
		string initfile;
	public:
	newalgo(CSR<INDEXTYPE, VALUETYPE> &A_csr, string input, string outputd, int init, double weight, double th, string ifile);
	void randInit();
	void initDFS();

	VALUETYPE frmodel(Coordinate<VALUETYPE> ci, Coordinate<VALUETYPE> cj){
		VALUETYPE dx = ci.x - cj.x;
        	VALUETYPE dy = ci.y - cj.y;
        	return -1.0 / (dx * dx + dy * dy);
	}
	vector<VALUETYPE> batchlayout(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);
	vector<VALUETYPE> EfficientVersion(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE);	
	void print();
	void writeRepulsiveForce(vector<Coordinate<VALUETYPE> > &repulse, string f);
	void writeToFileBH(Coordinate<VALUETYPE> *tCoordinates, string f);
	void writeToFile(string f);
};
#endif
