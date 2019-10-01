#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <random>
#include <utility>
#include "../sample/MortonCode.h"
#include "../sample/algorithms.h"
#include "../sample/BarnesHut.h"
using namespace std;

#define EPS 1.0e-7
#define VALUETYPE double
#define INDEXTYPE int

vector<pair<VALUETYPE, VALUETYPE> > readInput(string initfile){
	vector<pair<VALUETYPE, VALUETYPE> > input;
	VALUETYPE x, y;
        INDEXTYPE i;
	FILE *infile;
	infile = fopen(initfile.c_str(), "r");
	if(infile == NULL){
		cout << "ERROR in input coordinates file!\n" << endl;
		exit(1);
	}else{
		int index = 0;
		char line[256];
		VALUETYPE x, y;
		INDEXTYPE i;
		while(fgets(line, 256, infile)){
			sscanf(line, "%lf %lf %d", &x, &y, &i);
			input.push_back(make_pair(x, y)); 
			index++;
		}
	}
	fclose(infile);
	return input;
}
bool test(vector<pair<double, double> > a, vector<pair<double, double> > b){
	double diff = 0;
	for(int i = 0; i < a.size(); i++){
		diff += fabs(a[i].first - b[i].first) + fabs(a[i].second - b[i].second);
	}
	printf("Difference: %lf\n", diff);
	if(diff > a.size() * EPS){
		return false;
	}else{
		return true;
	}
}
void uTestAlgorithms(char *argv[]){
	CSR<INDEXTYPE, VALUETYPE> A_csr;
	string inputfile = "./datasets/input/3elt_dual.mtx";
        string outputdir = "./datasets/output/";
	SetInputMatricesAsCSR(A_csr, inputfile);
        A_csr.Sorted();
	vector<VALUETYPE> outputvec;
	algorithms algo = algorithms(A_csr, inputfile, outputdir, 0, 1, 1.2, "");
	algorithms algo2 = algorithms(A_csr, inputfile, outputdir, 0, 1, 1.2, "");
	outputvec = algo.cacheBlockingminiBatchForceDirectedAlgorithm(400, omp_get_max_threads(), 256, 0);
	outputvec = algo2.cacheBlockingminiBatchForceDirectedAlgorithmSD(400, omp_get_max_threads(), 256, 0);
	vector<pair<double, double> > first = readInput(outputdir+"3elt_dual.mtxCACHEMINB256PARAOUT400.txt");
	vector<pair<double, double> > second = readInput(outputdir+"3elt_dual.mtxCACHESDMINB256PARAOUT400.txt");
	if(test(first, second)){
		printf("OK!\n");
	}else{
		printf("Not OK!\n");
	}
}
int main(int argc, char* argv[]){
	
	uTestAlgorithms(argv);
        return 0;
}
