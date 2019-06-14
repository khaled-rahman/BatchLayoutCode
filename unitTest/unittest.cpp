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


#include "../sample/algorithms.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int

void TestCSRio(CSR<INDEXTYPE, VALUETYPE> &A_csr, char* argv[]){
        string fname = argv[1];
        //printCSR(A_csr);
        algorithms algoInit = algorithms(A_csr, 1.0, 1.0, fname);
	algorithms algoS = algorithms(A_csr, 1.0, 1.0, fname);
	algorithms algoSeq = algorithms(A_csr, 1.0, 1.0, fname);
        algorithms algo = algorithms(A_csr, 1.0, 1.0, fname);
	algorithms algoMini = algorithms(A_csr, 1.0, 1.0, fname);
        INDEXTYPE NUMOFTHREADS = atoi(argv[2]);
        INDEXTYPE ITERATIONS = atoi(argv[3]);
        for(INDEXTYPE i = 0; i < A_csr.rows; i++){
                algo.nCoordinates[i] = algoInit.nCoordinates[i];
		algoS.nCoordinates[i] = algoInit.nCoordinates[i];
		algoSeq.nCoordinates[i] = algoInit.nCoordinates[i];
		algoMini.nCoordinates[i] = algoInit.nCoordinates[i];
        }
	algoS.seqForceDirectedAlgorithm(ITERATIONS);
	algoSeq.seqAdjForceDirectedAlgorithm(ITERATIONS);
        algoMini.FinalminiBatchForceDirectedAlgorithmOutRand(ITERATIONS, NUMOFTHREADS, 256);
	for(int i = 2; i <= 32; i += 2)
		algo.cacheBlockingminiBatchForceDirectedAlgorithmRand(ITERATIONS, NUMOFTHREADS, 256, i);
}
void TestCSCio(CSC<INDEXTYPE, VALUETYPE> &A_csc, char* argv[]){
	string fname = "testing";
	printCSC(A_csc);
	algorithms algoInit = algorithms(A_csc, 1.0, 1.0, fname);
        algorithms algo = algorithms(A_csc, 1.0, 1.0, fname);
	INDEXTYPE NUMOFTHREADS = atoi(argv[2]);
	INDEXTYPE ITERATIONS = atoi(argv[3]);
	for(INDEXTYPE i = 0; i < A_csc.cols; i++){
        	algo.nCoordinates[i] = algoInit.nCoordinates[i];
        }
	//algo.cacheBlockingminiBatchForceDirectedAlgorithmRand(ITERATIONS, NUMOFTHREADS, 32);
}

void TestCoordinate(CSR<INDEXTYPE, VALUETYPE> &A_csr){
	Coordinate<VALUETYPE> *nCoordinates, *prevCoordinates;
	nCoordinates = new Coordinate<VALUETYPE>[A_csr.rows];
	prevCoordinates = new Coordinate<VALUETYPE>[A_csr.rows];
	for(INDEXTYPE i = 0; i < A_csr.rows; i++){
		nCoordinates[i] = Coordinate <VALUETYPE>(get_random<VALUETYPE>(-5.0, 5.0), get_random<VALUETYPE>(-5.0, 5.0));
		prevCoordinates[i] = Coordinate <VALUETYPE>(get_random<VALUETYPE>(-5.0, 5.0), get_random<VALUETYPE>(-5.0, 5.0)); 
	}
	
	for(INDEXTYPE i = 0; i < A_csr.rows; i++){
                cout << "Node:" << i << ", X:" << nCoordinates[i].getX() << ", Y:" << nCoordinates[i].getY()<<endl ;
        }
	cout << "getMagnitude:" << nCoordinates[0].getMagnitude() << endl;
	cout << "getDistance:" << nCoordinates[0].getDistance(nCoordinates[1]) << endl;
	cout << "getUnitVector:" << nCoordinates[0].getUnitVector().getX() << "," << nCoordinates[0].getUnitVector().getY() << endl;
}
void TestInitialization(CSR<INDEXTYPE, VALUETYPE> &A_csr, INDEXTYPE NUMOFTHREADS, INDEXTYPE ITERATIONS, string fname){
        algorithms algoInit1 = algorithms(A_csr, 1.0, 1.0, fname);
	//algoInit1.print();
	algorithms algoInit2 = algorithms(A_csr, 1.0, 1.0, fname);
        //algoInit2.initDFS();
	//algoInit2.print();
	//vector<VALUETYPE> seqout1 = algoInit1.seqForceDirectedAlgorithm(ITERATIONS);
	vector<VALUETYPE> seqout2;
	for(int i = 7; i < 8; i++){
		seqout2 = algoInit2.FinalminiBatchForceDirectedAlgorithmOutRand(ITERATIONS, NUMOFTHREADS, i+1);
		algoInit2.filename = fname;
	}
        /*vector<VALUETYPE> seqout2 = algoInit2.approxForceDirectedAlgorithm(ITERATIONS, NUMOFTHREADS, 8);
        printf("Iterations = %d, RandomInit = %lf, DFSInit = %lf\n", ITERATIONS, seqout1[0], seqout2[0]);
	string avgfile = "GraphAnalyticsProject.txt";
        ofstream output;
        output.open(avgfile, ofstream::app);
        output << ITERATIONS << "\t" << NUMOFTHREADS << "\t";
	output << seqout1[0] << "\t" << seqout2[0] << "\t" << seqout1[1] << "\t" << seqout2[1] << endl;
        output.close();
	*/
}


void TestAlgorithms(CSR<INDEXTYPE, VALUETYPE> &A_csr, INDEXTYPE NUMOFTHREADS, INDEXTYPE ITERATIONS, string filename){
	cout << "Total Number of iterations is:" << ITERATIONS << "\nPlease wait ... ... ...\n"<< endl;
	string fname = filename;
	algorithms algoInit = algorithms(A_csr, 1.0, 1.0, fname);
	algorithms algo = algorithms(A_csr, 1.0, 1.0, fname);
	algorithms algoMini = algorithms(A_csr, 1.0, 1.0, fname);
	//algo.print();
	//Coordinate<VALUETYPE> co = algo.calcAttraction(0, 1);
	//cout << "Coordinate returned:" << co.getX() <<"," << co.getY() << endl;
	//co = co + co.getUnitVector();
	//cout << "F-unitvector:" << co.getUnitVector().getX() << "," << co.getUnitVector().getY() << endl;
	printf("Number of threads:%d\n", NUMOFTHREADS);
	//algorithms algoTest = algorithms(A_csr, 1.0, 1.0, fname);
	//algoTest.seqAdjForceDirectedAlgorithm(ITERATIONS);
	
	vector<int> batchsize = {1, 256};
	vector<VALUETYPE> avgEnergy(batchsize.size(), 0.0), avgTime(batchsize.size(), 0.0);
	INDEXTYPE RUNS = 1;
	for(INDEXTYPE iter = 1; iter <= RUNS; iter += 1){
		for(INDEXTYPE i = 0; i < A_csr.rows; i++){
                                algo.nCoordinates[i] = algoInit.nCoordinates[i];
		}
		//vector<VALUETYPE> seqout = algo.seqForceDirectedAlgorithm(ITERATIONS);
		//vector<VALUETYPE> seqout = algo.seqAdjForceDirectedAlgorithm(ITERATIONS);
		vector<VALUETYPE> seqout = algo.FinalminiBatchForceDirectedAlgorithmOutRand(ITERATIONS, NUMOFTHREADS, 256);
		avgEnergy[0] += seqout[0];
		avgTime[0] += seqout[1];
		algo.filename = filename;
		for(INDEXTYPE b = 1; b < batchsize.size(); b++){
			for(INDEXTYPE i = 0; i < A_csr.rows; i++){
				algoMini.nCoordinates[i] = algoInit.nCoordinates[i];
			}
			//vector<VALUETYPE> output = algoMini.FinalminiBatchForceDirectedAlgorithmOutRand(ITERATIONS, NUMOFTHREADS, batchsize[b]);
                	vector<VALUETYPE> output = algoMini.cacheBlockingminiBatchForceDirectedAlgorithmRand(ITERATIONS, NUMOFTHREADS, 256, batchsize[b]); //here batchsize[b] is cacheblocksize
			avgEnergy[b] += output[0];
			avgTime[b] += output[1];
			algoMini.filename = filename;
		}
		
	}
	//string avgfile = "AvgResults_af_shell10.mtx_af_shell10.mtx_iter800.txt";
	string avgfile = "AvgResults_cbuckle.mtx_d_pretok_forcedistribution.txt";
        ofstream output;
       	output.open(avgfile, ofstream::app);
       	output << ITERATIONS << "\t" << NUMOFTHREADS << "\t";
	for(INDEXTYPE b = 0; b < batchsize.size(); b++)
		output << avgEnergy[b]/RUNS << "\t";
        for(INDEXTYPE b = 0; b < batchsize.size(); b++)
                output << avgTime[b]/RUNS << "\t";
        output << endl;
	output.close();
}
//#pragma omp declare reduction(plus:Coordinate<VALUETYPE>:omp_out += omp_in) initializer(omp_priv(omp_orig))
int main(int argc, char* argv[]){
        vector<int> tnums;
        tnums = {1, 2, 4, 8, 16, 32, 64};

        CSR<INDEXTYPE, VALUETYPE> A_csr;
	CSC<INDEXTYPE, VALUETYPE> A_csc;
        
	SetInputMatricesAsCSR(A_csr, argv);
        A_csr.Sorted();
	
	//SetInputMatricesAsCSC(A_csc, argv);
	//A_csc.Sorted();
	
	cout << "Entering TestCSR ... " << endl;
	//TestCSRio(A_csr, argv);
	//TestCSCio(A_csc, argv);
	//cout << "Entering TestCoordinate ..." << endl;
	//TestCoordinate(A_csr);
	/*int tt = 100;
	Coordinate<VALUETYPE> *nc = new Coordinate<VALUETYPE>[tt];
	for(int i=0; i<tt; i++){
		nc[i] = Coordinate <VALUETYPE>(i*1.0,i*1.0);
	}
	Coordinate<VALUETYPE> f = Coordinate<VALUETYPE>(0, 0);
	//#pragma omp declare reduction(plus:Coordinate<VALUETYPE>:omp_out += omp_in) initializer(omp_priv(omp_orig))
	omp_set_num_threads(8);
	#pragma omp parallel for reduction(plus:f)
	for(int i=0; i<tt; i++){
		f += nc[i];
	}
	
	cout << "f-x:" << f.getX() << ", f-y:" << f.getY() << endl;
	*/
	//cout << "Entering TestAlgorithms ..." << endl;
	INDEXTYPE NUMOFTHREADS, ITERATIONS;
	NUMOFTHREADS = atoi(argv[2]);
	ITERATIONS = atoi(argv[3]);
	TestAlgorithms(A_csr, NUMOFTHREADS, ITERATIONS, argv[1]);
	//TestInitialization(A_csr, NUMOFTHREADS, ITERATIONS, argv[1]);
        return 0;
}
