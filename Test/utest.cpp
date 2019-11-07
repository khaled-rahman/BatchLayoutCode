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
#include "../sample/newalgo.h"
#include "../sample/BarnesHut.h"
using namespace std;

#define EPS 2.2e-16
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
void test(Coordinate<VALUETYPE> *a, Coordinate<VALUETYPE> *b, int size){
	double maxdiff = 2 * 10 * 1000 * size * EPS;
	int errc = 0;
	for(int i = 0; i <size; i++){
		double diff = fabs(a[i].x - b[i].x);
		if(diff > maxdiff) errc++;
		diff = fabs(a[i].y - b[i].y);
		if(diff > maxdiff) errc++;
	}
	if(errc == 0){
		printf("Passed Test!\n");
	}else{
		printf("Not passed! Total error = %d out of %d. Error rate = %lf\n", errc, 2 * size, 1.0 * errc/ (2.0 * size));
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
}

void uTestNewAlgo(int argc, char *argv[]){
	CSR<INDEXTYPE, VALUETYPE> A_csr;
        //string inputfile = "./datasets/input/jagmesh1.mtx";
	 string inputfile = "./datasets/input/3elt_dual.mtx";
	//string inputfile = "./datasets/input/skirt.mtx";
        string outputdir = "./datasets/output/";
        SetInputMatricesAsCSR(A_csr, inputfile);
        A_csr.Sorted();
        vector<VALUETYPE> outputvec;
   #if 0
        algorithms algo = algorithms(A_csr, inputfile, outputdir, 0, 1, 1.2, "");
	algo.cacheBlockingminiBatchForceDirectedAlgorithm(500, 48, 256, 0);	
        algorithms algo2 = algorithms(A_csr, inputfile, outputdir, 0, 1, 1.2, "");
	algo2.cacheBlockingminiBatchForceDirectedAlgorithmSD(500, 48, 256, 0);
        test(algo.nCoordinates, algo2.nCoordinates, A_csr.rows);	
   #endif	
	newalgo na = newalgo(A_csr, inputfile, outputdir, 0, 1, 1.2, "");
	na.EfficientVersionUnRoll(5, 48, 256);
        
        newalgo na2 = newalgo(A_csr, inputfile, outputdir, 0, 1, 1.2, "");
	na2.EfficientVersion(5, 48, 256);	
}
#define BATCHSIZE 48
#define ITERATIONS 500
void GetAvgTimes(int argc, char *argv[], int nrep, string inputfile=""){
   double timep0 = 0.0, timep1 = 0.0, time1=0.0, time2=0.0; 
   double energyp0 = 0.0, energyp1 = 0.0, energy1=0.0, energy2=0.0; 
   CSR<INDEXTYPE, VALUETYPE> A_csr;
   //string inputfile = "./datasets/input/af_shell10.mtx";
   //string inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/Flan_1565.mtx";
   //string inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/netherlands_osm.mtx";
   //string inputfile = "./datasets/input/3elt_dual.mtx";
   //string inputfile = "./datasets/input/pkustk02.mtx";
   //string inputfile = "./datasets/input/power.mtx";
   //string inputfile = "./datasets/input/fe_4elt2.mtx";
   //string inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/tube2.mtx";
   //string inputfile = "./datasets/input/pdb1HYS.mtx";
   //string inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/flGenForDiGraph/datasets/input/gridgena.mtx";
   //string inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/flGenForDiGraph/datasets/input/finan512.mtx";
   //string inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/luxembourg_osm.mtx";
   //string inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/com-Youtube.mtx";
   string outputdir = "./datasets/output/";
   SetInputMatricesAsCSR(A_csr, inputfile);
   A_csr.Sorted();
   vector<VALUETYPE> outputvec;
 /*
  *   Original implementation
  */
   algorithms algo = algorithms(A_csr, inputfile, outputdir, 0, 1, 1.2, "");
   
   algorithms algo1 = algorithms(A_csr, inputfile, outputdir, 0, 1, 1.2, "");
   //algo1.cacheBlockingminiBatchForceDirectedAlgorithm(1, 48, 64, 0); 
/*
 * To compare two versions 
 */
   newalgo na = newalgo(A_csr, inputfile, outputdir, 0, 1, 1.2, "");
   newalgo na2 = newalgo(A_csr, inputfile, outputdir, 0, 1, 1.2, "");

   for (int i=0; i < nrep; i++)
   {
      //outputvec = algo1.BarnesHutApproximation(1, 48, BATCHSIZE, 1.2, 0);
      //timep1 += outputvec[1];
      //energyp1 += (outputvec[0]/nrep);
      //algo1.filename = inputfile;

      //outputvec = na.EfficientVersion(5, 18, 256);
      //outputvec = na.EfficientVersion(5, 18, 64);
      //outputvec = algo.cacheBlockingminiBatchForceDirectedAlgorithm(1, 10, BATCHSIZE, 0);
      //timep0 += outputvec[1];
      //energyp0 += (outputvec[0]/nrep);      
      //algo.filename = inputfile;
      
      //outputvec = na.EfficientVersion(ITERATIONS, 48, BATCHSIZE);
      //na.filename = inputfile;
      //time1 += outputvec[1];
      //energy1 += (outputvec[0]/nrep); // to avoid overflow 
      //cout << "1st energy=" << outputvec[0] << "  Time = " << outputvec[1] <<endl; 

      //outputvec = na2.EfficientVersionMdim(5, 18, 256);
      //outputvec = na2.EfficientVersionMdim(5, 18, 64);
      outputvec = na2.EfficientVersionMdim(ITERATIONS, 48, 48);
      na2.filename = inputfile;
      //cout << "2nd energy=" << outputvec[0] << " Time = " << outputvec[1] << endl; 
      time2 += outputvec[1];
      energy2 += (outputvec[0]/nrep); // to avoid overflow 
   }
   time1 /= nrep;
   time2 /= nrep;
   timep0 /= nrep;
   timep1 /= nrep;

   time1 /= ITERATIONS;
   time2 /= ITERATIONS;
   timep0 /= ITERATIONS;
   timep1 /= ITERATIONS;

   cout << "BarnesHutApprox: Avg time"<<"("<<nrep<<") = " << timep1 <<endl;
   cout << "BarnesHutApprox: Avg Energy"<<"("<<nrep<<") = " << energyp1 <<endl;

   cout << "BatchLayout: Avg time"<<"("<<nrep<<") = " << timep0 <<endl;
   cout << "BatchLayout: Avg Energy"<<"("<<nrep<<") = " << energyp0 <<endl;

   cout << "Efficient version: Avg time"<<"("<<nrep<<") = " << time1 <<endl;
   cout << "Efficient version: Avg Energy"<<"("<<nrep<<") = " << energy1 <<endl;
#if 0
   cout << "Efficient Unroll version: Avg time"<<"("<<nrep<<") = " << time2 <<endl;
   cout << "Efficient Unroll: Avg Energy"<<"("<<nrep<<") = " << energy2 <<endl;
   cout << "Speedup of Unroll version = " << time1/time2 << endl;
#else
   cout << "Efficient M-dim version: Avg time"<<"("<<nrep<<") = " << time2 <<endl;
   cout << "Efficient M-dim: Avg Energy"<<"("<<nrep<<") = " << energy2 <<endl;
   cout << "Speedup of M-dim version = " << time1/time2 << endl;
   cout << "Speedup of old version = " << time2/time1 << endl;
#endif

}

#define DEBUG 1 

void runall(int argc, char *argv[])
{       
        string inputfile = "./datasets/input/power.mtx";
        GetAvgTimes(argc, argv, 5, inputfile);
        inputfile = "./datasets/input/3elt_dual.mtx";
        GetAvgTimes(argc, argv, 5, inputfile);
        inputfile = "./datasets/input/pkustk02.mtx";
        GetAvgTimes(argc, argv, 5, inputfile);
        inputfile = "./datasets/input/fe_4elt2.mtx";
        GetAvgTimes(argc, argv, 5, inputfile);
        inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/tube2.mtx";
        GetAvgTimes(argc, argv, 5, inputfile);
        inputfile = "./datasets/input/pdb1HYS.mtx";
        GetAvgTimes(argc, argv, 5, inputfile);
        inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/flGenForDiGraph/datasets/input/gridgena.mtx";
        GetAvgTimes(argc, argv, 5, inputfile);
        inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/flGenForDiGraph/datasets/input/finan512.mtx";
        GetAvgTimes(argc, argv, 5, inputfile);
        inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/luxembourg_osm.mtx";
        GetAvgTimes(argc, argv, 2, inputfile);
        inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/com-Youtube.mtx";
        //GetAvgTimes(argc, argv, 2, inputfile);
        inputfile = "./datasets/input/af_shell10.mtx";
        //GetAvgTimes(argc, argv, 2, inputfile);
        inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/Flan_1565.mtx";
        GetAvgTimes(argc, argv, 1, inputfile);
}

int main(int argc, char* argv[]){
	
	//uTestAlgorithms(argv);
        //uTestNewAlgo(argc, argv);
   #if DEBUG == 1
	//runall(argc, argv);
	//string inputfile = "/N/u/morahma/Research/Spring2019/FirstForceDirectedGraph/ParallelGraphVis/SMU/datasets/input/Flan_1565.mtx";
        //string inputfile = "./datasets/input/af_shell10.mtx";
	string inputfile = "./datasets/input/3elt_dual.mtx";
	GetAvgTimes(argc, argv, 1, inputfile); 
   #else
	//runall(argc, argv);
        //inputfile = "./datasets/input/3elt_dual.mtx";
        //GetAvgTimes(argc, argv, 5, inputfile);
   #endif
	return 0;
}
