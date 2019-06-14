#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
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

#include "../utility.h"
#include "../CSR.h"
#include "../multiply.h"

#include "../sample/commonutility.hpp"
#include "../sample/Coordinate.hpp"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int
#define CACHEBLOCK 4
#define MAXMIN 3.0
#define t 0.999
#define PI 3.14159265358979323846
static int PROGRESS = 0;

Coordinate<VALUETYPE> plusd(Coordinate<VALUETYPE> omp_in, Coordinate<VALUETYPE> omp_out){return omp_in + omp_out;}

#pragma omp declare reduction(plus:Coordinate<VALUETYPE>:omp_out += omp_in) initializer(omp_priv = Coordinate<VALUETYPE>(0.0, 0.0))

#pragma omp declare reduction(vplus:vector<Coordinate<VALUETYPE> > : \
                             transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), plusd)) \
                    initializer(omp_priv = omp_orig)
class algorithms{
	public:
		CSR<INDEXTYPE, VALUETYPE> graph;
		CSC<INDEXTYPE, VALUETYPE> graphCSC;
		vector<Coordinate<VALUETYPE> > nCoordinates, prevCoordinates;
		VALUETYPE K, C;
		string filename;
	public:
	algorithms(CSR<INDEXTYPE, VALUETYPE> &A_csr, VALUETYPE c, VALUETYPE k, string f){
		graph.make_empty();
		graph = A_csr;
		K = k;
		C = c;
		filename = f;
		nCoordinates.resize(A_csr.rows);
        	prevCoordinates.resize(A_csr.rows);
		initDFS();
        	/*for(INDEXTYPE i = 0; i < A_csr.rows; i++){
                	nCoordinates[i] = Coordinate <VALUETYPE>(get_random<VALUETYPE>(-MAXMIN, MAXMIN), get_random<VALUETYPE>(-MAXMIN, MAXMIN));
                	prevCoordinates[i] = Coordinate <VALUETYPE>(get_random<VALUETYPE>(0.0, 0.0), get_random<VALUETYPE>(0.0, 0.0));
       		 }*/
	}
	algorithms(CSC<INDEXTYPE, VALUETYPE> &A_csc, VALUETYPE c, VALUETYPE k, string f){
                graphCSC.make_empty();
                graphCSC = A_csc;
                K = k;
                C = c;
                filename = f;
                //nCoordinates = new Coordinate<VALUETYPE>[A_csc.cols];
                //prevCoordinates = new Coordinate<VALUETYPE>[A_csc.cols];
                nCoordinates.resize(A_csc.cols);
                prevCoordinates.resize(A_csc.cols);
		for(INDEXTYPE i = 0; i < A_csc.cols; i++){
                        nCoordinates[i] = Coordinate <VALUETYPE>(get_random<VALUETYPE>(-MAXMIN, MAXMIN), get_random<VALUETYPE>(-MAXMIN, MAXMIN));
                        prevCoordinates[i] = Coordinate <VALUETYPE>(get_random<VALUETYPE>(0.0, 0.0), get_random<VALUETYPE>(0.0, 0.0));
                }
        }
	~algorithms(){
		//free(nCoordinates);
		//free(prevCoordinates);
	}
	void initDFS(){
                int visited[graph.rows] = {0};
                stack <int> STACK;
		double minX, minY, maxX, maxY, scalefactor = 1.0;
		minX = minY = 1.0;//numeric_limits<double>::max();
		maxX = maxY = 1.0;//numeric_limits<double>::min();
                STACK.push(0);
		double radi = 0.1;
                visited[0] = 1;
                nCoordinates[0] = Coordinate <VALUETYPE>(1.0, 1.0);
                while(!STACK.empty()){
                        int parent = STACK.top();
                        STACK.pop();
                        if(parent < graph.rows - 1 && (graph.rowptr[parent+1] - graph.rowptr[parent]) > 0){
                                double deg = 360.0 / (graph.rowptr[parent+1] - graph.rowptr[parent]);
                                double degree = 0;
                                for(INDEXTYPE n = graph.rowptr[parent]; n < graph.rowptr[parent+1]; n++){
                                        if(visited[graph.colids[n]] == 0){
                                                nCoordinates[graph.colids[n]] = Coordinate <VALUETYPE>(nCoordinates[parent].getX() + radi*cos(PI*(degree)/180.0), nCoordinates[parent].getY() + radi*sin(PI*(degree)/180.0));
                                                visited[graph.colids[n]] = 1;
                                                STACK.push(graph.colids[n]);
                                                degree += deg;
						if(minX > nCoordinates[graph.colids[n]].getX()){
							minX = nCoordinates[graph.colids[n]].getX();
						}
						else if(maxX < nCoordinates[graph.colids[n]].getX()){
                                                        maxX = nCoordinates[graph.colids[n]].getX();
                                                }
						if(minY > nCoordinates[graph.colids[n]].getY()){
                                                        minY = nCoordinates[graph.colids[n]].getY();
                                                }
                                                else if(maxY < nCoordinates[graph.colids[n]].getY()){
                                                        maxY = nCoordinates[graph.colids[n]].getY();
                                                }
                                        }
                                }
                        }else{
				if((graph.nnz - graph.rowptr[parent]) <= 0) continue;

                                double deg = 360.0 / (graph.nnz - graph.rowptr[parent]);
                                double degree = 0;
                                for(INDEXTYPE n = graph.rowptr[parent]; n < graph.nnz; n++){
                                        if(visited[graph.colids[n]] == 0){
                                                nCoordinates[graph.colids[n]] = Coordinate <VALUETYPE>(nCoordinates[parent].getX() + radi*cos(PI*(degree)/180.0), nCoordinates[parent].getY() + radi*sin(PI*(degree)/180.0));
                                                visited[graph.colids[n]] = 1;
                                                STACK.push(graph.colids[n]);
                                                degree += deg;
						if(minX > nCoordinates[graph.colids[n]].getX()){
                                                        minX = nCoordinates[graph.colids[n]].getX();
                                                }
                                                else if(maxX < nCoordinates[graph.colids[n]].getX()){
                                                        maxX = nCoordinates[graph.colids[n]].getX();
                                                }
                                                if(minY > nCoordinates[graph.colids[n]].getY()){
                                                        minY = nCoordinates[graph.colids[n]].getY();
                                                }
                                                else if(maxY < nCoordinates[graph.colids[n]].getY()){
                                                        maxY = nCoordinates[graph.colids[n]].getY();
                                                }
                                        }
                                }
                        }
                }
		scalefactor = 2.0 * MAXMIN / max(maxX - minX, maxY - minY);
		//printf("Scaling: %lf\n", scalefactor);
		for(int i = 0; i < graph.rows; i++){
			nCoordinates[i] = nCoordinates[i] * scalefactor;
		}
        }
	Coordinate<VALUETYPE> calcAttraction(INDEXTYPE i, INDEXTYPE j){
                Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
		for(INDEXTYPE n = graph.rowptr[i]; n < graph.rowptr[i+1]; n++){
			f = f + (this->nCoordinates[graph.colids[n]] - this->nCoordinates[i]) * ((this->nCoordinates[graph.colids[n]] - this->nCoordinates[i]).getMagnitude()/K) - calcRepulsion(i, graph.colids[n]);
		}
                return f;
        }

	Coordinate<VALUETYPE> calcAttraction2(INDEXTYPE i, INDEXTYPE j){
                Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
              
                for(INDEXTYPE n = graph.rowptr[i]; n < graph.rowptr[i+1]; n++){
                	INDEXTYPE nID = graph.colids[n];
                        f = f + (this->prevCoordinates[nID] - this->prevCoordinates[i]) * ((this->prevCoordinates[nID] - this->prevCoordinates[i]).getMagnitude()/K) - calcRepulsion(i, nID);
                        }
                return f;
        }	

	Coordinate<VALUETYPE> calcRepulsion(INDEXTYPE i, INDEXTYPE n){
		Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                if((this->nCoordinates[n] - this->nCoordinates[i]).getMagnitude2() > 0)
                	f = f - (this->nCoordinates[n] - this->nCoordinates[i]) * (C * K * K /(this->nCoordinates[n] - this->nCoordinates[i]).getMagnitude2());
		return f;
	}
	
	Coordinate<VALUETYPE> calcRepulsion2(INDEXTYPE i, INDEXTYPE n){
                Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                if((this->prevCoordinates[n] - this->prevCoordinates[i]).getMagnitude2() > 0)
                        f = f - (this->prevCoordinates[n] - this->prevCoordinates[i]) * (C * K * K /(this->prevCoordinates[n] - this->prevCoordinates[i]).getMagnitude2());
                return f;
        }

	VALUETYPE updateStepLength(VALUETYPE STEP, VALUETYPE ENERGY, VALUETYPE ENERGY0){
		/*if(ENERGY < ENERGY0){
			PROGRESS = PROGRESS + 1;
			if(PROGRESS >= 5){
				PROGRESS = 0;
				STEP = STEP / t;
			}
		}else{
			PROGRESS = 0;
			STEP = t * STEP;
		}*/
		return STEP * t;
	}
	
	//sequantial implementation of O(n^2) algorithm
	vector<VALUETYPE> seqForceDirectedAlgorithm(INDEXTYPE ITERATIONS){
		INDEXTYPE LOOP = 0;
		VALUETYPE start, end, ENERGY, ENERGY0;
		VALUETYPE STEP = 1.0;
		vector<VALUETYPE> result;
		ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
		start = omp_get_wtime();
		PROGRESS = 0;
		while(LOOP < ITERATIONS){
			ENERGY0 = ENERGY;
			ENERGY = 0;
			Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
			INDEXTYPE j;
			INDEXTYPE k;
			for(INDEXTYPE i = 0; i < graph.rows; i++){
				f = Coordinate <VALUETYPE>(0.0, 0.0);
				k = graph.rowptr[i];
				for(j = 0; j < graph.rows; j++){
                               		if(j == graph.colids[k]){
						//printf("Neighbors: %d - %d\n", i, j);
						f = f + (this->nCoordinates[j] - this->nCoordinates[i]) * ((this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude()/K);
                                                if(k < graph.rowptr[i+1] - 1){
							k++;
						}
					}
                               		else{
						//f = f + calcRepulsion(i, j);
						if((this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude2() > 0)
                                                        f = f - (this->nCoordinates[j] - this->nCoordinates[i]) * (C * K * K /(this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude2());
					}
				}
				this->nCoordinates[i] = this->nCoordinates[i] + f.getUnitVector() * STEP;
                                ENERGY = ENERGY + f.getMagnitude2();
			}
			STEP = updateStepLength(STEP, ENERGY, ENERGY0);
			LOOP++;
		}
		end = omp_get_wtime();
		cout << "Energy:" << ENERGY << endl;
		printf("Energy:%lf\n", ENERGY);
		cout << "Sequential Wall time required:" << end - start << endl;
		result.push_back(ENERGY);
		result.push_back(end - start);
		writeToFile("SEQU" + to_string(ITERATIONS));
		return result;
	}
	
	vector<VALUETYPE> seqAdjForceDirectedAlgorithm(INDEXTYPE ITERATIONS){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
		/*vector<vector<int> > adjmat(graph.rows, vector<int>(graph.rows, 0));
		for(INDEXTYPE i = 0; i < graph.rows; i++){
			for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j++){
				adjmat[i][graph.colids[j]] = 1;
			}
		}*/
		vector<int> adjvect(graph.rows, 0);
                vector<VALUETYPE> result;
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                start = omp_get_wtime();
                PROGRESS = 0;
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                        INDEXTYPE j;
                        INDEXTYPE k;
                        for(INDEXTYPE i = 0; i < graph.rows; i++){
                                f = Coordinate <VALUETYPE>(0.0, 0.0);
				for(j = graph.rowptr[i]; j < graph.rowptr[i+1]; j++){
					adjvect[graph.colids[j]] = 1;
				}
                                for(j = 0; j < graph.rows; j++){
                                        if(adjvect[j] == 1){
						adjvect[j] = 0;
						//printf("Neighbors: %d - %d\n", i, j);
                                                f = f + (this->nCoordinates[j] - this->nCoordinates[i]) * ((this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude()/K);
                                        }
                                        else{
                                                if((this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude2() > 0)
                        				f = f - (this->nCoordinates[j] - this->nCoordinates[i]) * (C * K * K /(this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude2());
                                        }
                                }
                                this->nCoordinates[i] = this->nCoordinates[i] + f.getUnitVector() * STEP;
                                ENERGY = ENERGY + f.getMagnitude2();
                        }
                        STEP = updateStepLength(STEP, ENERGY, ENERGY0);
                        LOOP++;
                }
                end = omp_get_wtime();
                cout << "Energy Adj:" << ENERGY << endl;
		printf("Energy:%lf\n", ENERGY);
                cout << "Sequential Adj Wall time required:" << end - start << endl;
                result.push_back(ENERGY);
                result.push_back(end - start);
                writeToFile("SEQUADJ" + to_string(ITERATIONS));
                return result;
        }
	
	//benign race parallel implementation of O(n^2) outer loop parallelization
	vector<VALUETYPE> para2ForceDirectedAlgorithmOut(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
		vector<VALUETYPE> result;
                ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                PROGRESS = 0;
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        //prevCoordinates = new Coordinate<VALUETYPE>[graph.rows];
                        #pragma omp parallel for reduction(+:ENERGY)
                        for(INDEXTYPE i = 0; i < graph.rows; i++){
                                Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                                INDEXTYPE k = graph.rowptr[i];
                                for(INDEXTYPE j = 0; j < graph.rows; j++){
                                        if(j == graph.colids[k] && k < graph.nnz){
                                                f += (nCoordinates[j] - nCoordinates[i]) * ((nCoordinates[j] - nCoordinates[i]).getMagnitude()/K);
                                                if(k < graph.rowptr[i+1]-1){
                                                        k++;
                                                }
                                        }
                                        else{
                                                f += calcRepulsion(i, j);
                                        }
                                }
				#pragma omp critical
                                nCoordinates[i] = nCoordinates[i] + f.getUnitVector() * STEP;
                                ENERGY += f.getMagnitude2();
                        }
                        STEP = updateStepLength(STEP, ENERGY, ENERGY0);
                        LOOP++;
                }
                end = omp_get_wtime();
                cout << "Energy2(outer):" << ENERGY << endl;
		result.push_back(ENERGY);
                cout << "Parallel2(outer) Wall time required:" << end - start << endl;
		result.push_back(end - start);
                writeToFile("PARA2OUT" + to_string(ITERATIONS));
		return result;
        }
	
		
	//benign race minibatchparallel implementation of O(n^2) outer loop parallelization
	vector<VALUETYPE> miniBatchForceDirectedAlgorithmOut(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
		vector<VALUETYPE> result;
                ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                PROGRESS = 0;
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
			//prevCoordinates = new Coordinate<VALUETYPE>[graph.rows];
			for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                        	#pragma omp parallel for reduction(+:ENERGY)
                        	for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
					if(i >= graph.rows) continue;
					Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                                	INDEXTYPE k = graph.rowptr[i];
                                	for(INDEXTYPE j = 0; j < graph.rows; j++){
                                        	if(j == graph.colids[k] && k < graph.nnz){
                                                	f += (nCoordinates[j] - nCoordinates[i]) * ((nCoordinates[j] - nCoordinates[i]).getMagnitude()/K);
                                                        if(k < graph.rowptr[i+1]-1){
                                                        	k++;
                                                	}
                                        	}else{
                                                	f += calcRepulsion(i, j);
                                        	}
                                	}
					nCoordinates[i] += f.getUnitVector() * STEP;
                                        ENERGY += f.getMagnitude2();
                        	}
			}
                        STEP = updateStepLength(STEP, ENERGY, ENERGY0);
                        LOOP++;
                }
                end = omp_get_wtime();
		cout << "Batch Size:" << BATCHSIZE << endl;
                cout << "Minbatch Energy:" << ENERGY << endl;
		result.push_back(ENERGY);
                cout << "Minibatch Parallel(inner) Wall time required:" << end - start << endl;
		result.push_back(end-start);
		writeToFile("MINPARAOUT" + to_string(ITERATIONS));
		return result;
        }
		
	//O(n^2) for random order of vertices
	vector<VALUETYPE> FinalminiBatchForceDirectedAlgorithmOutRand(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
		srand(unsigned(time(0)));
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0, ATTRACTIVEENERGY;
                VALUETYPE STEP = 1.0;// - log2(1.0 * BATCHSIZE) / 16.0;
                vector<VALUETYPE> result;
                vector<INDEXTYPE> indices;
                for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                PROGRESS = 0;
		double FLOPS = 0.0;
		//LOOP < ITERATIONS
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
			ATTRACTIVEENERGY = 0;
			FLOPS = 0.0;
			//uncomment the below line to calculate enery in random vertex order
                        //random_shuffle(indices.begin(), indices.end());
       			#pragma omp parallel                 
			for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                                #pragma omp for schedule(static) reduction(+:ENERGY,FLOPS,ATTRACTIVEENERGY)	
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
					Coordinate<VALUETYPE> fattract = Coordinate <VALUETYPE>(0.0, 0.0);
                                        INDEXTYPE k = graph.rowptr[indices[i]];
					#pragma omp simd
                                        for(INDEXTYPE j = 0; j < graph.rows; j++){
						if(j == graph.colids[k]){
							//printf("Neighbors: %d - %d\n", i, j);
                                                	f += (nCoordinates[j] - nCoordinates[indices[i]]) * ((nCoordinates[j] - nCoordinates[indices[i]]).getMagnitude()/K);
							//fattract += (nCoordinates[j] - nCoordinates[indices[i]]) * ((nCoordinates[j] - nCoordinates[indices[i]]).getMagnitude()/K);
                                                        if(k < graph.rowptr[indices[i]+1]-1){
                                                                k++;
                                                        }
						}else{
                                                        if((this->nCoordinates[j] - this->nCoordinates[indices[i]]).getMagnitude2() > 0)
                        				{
								f = f - (this->nCoordinates[j] - this->nCoordinates[indices[i]]) * (C * K * K /(this->nCoordinates[j] - this->nCoordinates[indices[i]]).getMagnitude2());
								fattract += (this->nCoordinates[j] - this->nCoordinates[indices[i]]) * (C * K * K /(this->nCoordinates[j] - this->nCoordinates[indices[i]]).getMagnitude2());
							}
                                                }
                                        }
                                        prevCoordinates[indices[i]] = f;
                                        ENERGY += f.getMagnitude2();
					ATTRACTIVEENERGY += fattract.getMagnitude2();
					//printf("Iteration = %d, Batch id = %d, Node = %d\n", LOOP, b, indices[i]);
					//printf("prevCor[%d].Unit : x = %lf, y = %lf\n", i, prevCoordinates[i].getUnitVector().getX(), prevCoordinates[i].getUnitVector().getY());
				}
				#pragma omp for schedule(static) reduction(+:FLOPS)  
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[indices[i]] = nCoordinates[indices[i]] + prevCoordinates[indices[i]].getUnitVector() * STEP;
                                }
                        }
			//printf("Iteration = %d, Energy = %lf\n", LOOP, ENERGY);
			STEP = updateStepLength(STEP, ENERGY, ENERGY0);
                        LOOP++;
			//print();
                }
		end = omp_get_wtime();
                cout << "Final(rand) Batch Size:" << BATCHSIZE << endl;
                cout << "Final(rand) Minbatch Energy:" << ENERGY << endl;
                result.push_back(ENERGY);
                cout << "Final(rand) Minibatch Parallel(inner) Wall time required:" << end - start << endl;
                result.push_back(ATTRACTIVEENERGY);
                writeToFile("MINB"+ to_string(BATCHSIZE)+"PARAOUTRAND" + to_string(ITERATIONS));
                return result;
	}
	vector<VALUETYPE> cacheBlockingminiBatchForceDirectedAlgorithmRand(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, INDEXTYPE CBLOCK){
                srand(unsigned(time(0)));
                INDEXTYPE LOOP = 0;
		INDEXTYPE blocky = CBLOCK, blockx = 2;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;// - log2(1.0 * BATCHSIZE) / 16.0;
                vector<VALUETYPE> result;
                vector<INDEXTYPE> indices;
                for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                PROGRESS = 0;
		while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
			for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
				#pragma omp parallel for schedule(static) //reduction(vplus:prevCoordinates)
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += blockx){
                                	vector<int> kindex(blockx, 0);
					for(INDEXTYPE bi = 0; bi < blockx && i + bi < (b + 1) * BATCHSIZE; bi++){
						if(i+bi >= graph.rows) continue;
						kindex[bi] = graph.rowptr[i+bi];
						prevCoordinates[i+bi] = Coordinate <VALUETYPE>(0.0, 0.0);
					}
                                	for(INDEXTYPE j = 0; j < graph.rows; j += blocky){
						for(INDEXTYPE bi = 0; bi < blockx && i + bi < (b + 1) * BATCHSIZE; bi++){
							if(i+bi >= graph.rows) continue;
							Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                                        		for(INDEXTYPE bj = 0; bj < blocky && j + bj < graph.rows; bj++){
								if(j + bj == graph.colids[kindex[bi]]){
									//printf("Neighbors: %d - %d\n", i+bi, j+bj);
									f += (nCoordinates[j+bj] - nCoordinates[i+bi]) * ((nCoordinates[j+bj] - nCoordinates[i+bi]).getMagnitude()/K);
									if(kindex[bi] < graph.rowptr[i+bi+1] - 1){
										kindex[bi]++;
									}
								}else{
									if((this->nCoordinates[j+bj] - this->nCoordinates[i+bi]).getMagnitude2() > 0)
                                                        		{
                                                                		f = f - (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]) * (C * K * K /(this->nCoordinates[j+bj] - this->nCoordinates[i+bi]).getMagnitude2());
                                                        		}
								}
							}
							prevCoordinates[i+bi] += f;
							
						}
					}
					//printf("prevCor[%d].Unit : x = %lf, y = %lf\n", i, prevCoordinates[i].getUnitVector().getX(), prevCoordinates[i].getUnitVector().getY());
				}
                                #pragma omp parallel for schedule(static) reduction(+:ENERGY)  
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[i] = nCoordinates[i] + prevCoordinates[i].getUnitVector() * STEP;
                                	ENERGY += prevCoordinates[i].getMagnitude2();
				}
			}
			//printf("Iteration = %d, Energy = %lf\n", LOOP, ENERGY);
                        STEP = updateStepLength(STEP, ENERGY, ENERGY0);
                        LOOP++;
			//print();
                }
                end = omp_get_wtime();
                cout << "Final(Cache) Batch Size:" << BATCHSIZE << ", BLOCKSIZE:" << CBLOCK << endl;
                cout << "Final(Cache) Minbatch Energy:" << ENERGY << endl;
                result.push_back(ENERGY);
                cout << "Final(Cache) Minibatch Parallel(inner) Wall time required:" << end - start << endl;
                result.push_back(end - start);
                writeToFile("CACHEMINB"+ to_string(BATCHSIZE)+"PARAOUTRAND" + to_string(ITERATIONS));
                return result;
        }
	vector<VALUETYPE> approxForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0, approxITER = 100;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
		stack <int> STACKnode;
                vector<VALUETYPE> result;
		vector<INDEXTYPE> indices;
                for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
		omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                PROGRESS = 0;
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                        INDEXTYPE j;
                        INDEXTYPE k;
			//random_shuffle(indices.begin(), indices.end());
                        for(INDEXTYPE b = 0; b < (int)ceil(1.0 * graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static) reduction(+:ENERGY)   
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
					if(LOOP < approxITER){
						INDEXTYPE k = graph.rowptr[indices[i]];
                                        	for(INDEXTYPE j = 0; j < graph.rows; j++){
                                                	if(j == graph.colids[k] && k < graph.nnz){
                                                        	f += (nCoordinates[j] - nCoordinates[indices[i]]) * ((nCoordinates[j] - nCoordinates[indices[i]]).getMagnitude()/K);
                                                        	if(k < graph.rowptr[indices[i]+1]-1){
                                                                	k++;
                                                        	}
                                                	}else{
								if((this->nCoordinates[j] - this->nCoordinates[indices[i]]).getMagnitude2() > 0)
                                                        	{
                                                                	f = f - (this->nCoordinates[j] - this->nCoordinates[indices[i]]) * (C * K * K /(this->nCoordinates[j] - this->nCoordinates[indices[i]]).getMagnitude2());
                                                        	}
                                                	}
                                        	}
					}
					else{
						unordered_map<int, int> neighbors;
                                        	neighbors.insert(pair<int, int>(indices[i], indices[i]));
						if(i < graph.rows - 1){
                        				for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j++){
								f += (nCoordinates[graph.colids[j]] - nCoordinates[indices[i]]) * ((nCoordinates[graph.colids[j]] - nCoordinates[indices[i]]).getMagnitude()/K);
								STACKnode.push(graph.colids[j]);
								neighbors.insert(pair<int, int>(graph.colids[j], indices[i]));
                        				}
                				}else{
                        				for(INDEXTYPE j = graph.rowptr[i]; j < graph.nnz; j++){
								f += (nCoordinates[graph.colids[j]] - nCoordinates[indices[i]]) * ((nCoordinates[graph.colids[j]] - nCoordinates[indices[i]]).getMagnitude()/K);
                                                        	STACKnode.push(graph.colids[j]);
                                                       		neighbors.insert(pair<int, int>(graph.colids[j], indices[i]));
                        				}
                				}
						int countNodes = 150;
						while(!STACKnode.empty()){
							int currentn = STACKnode.top();
							STACKnode.pop();
							for(INDEXTYPE n = graph.rowptr[currentn]; n < graph.rowptr[currentn+1]; n++){
								if(neighbors.count(graph.colids[n]) < 1){
									f += calcRepulsion(indices[i], graph.colids[n]);
									STACKnode.push(graph.colids[n]);
									countNodes--;
								}
							}
							if(countNodes <= 0)break;
						}
                                	}
					prevCoordinates[indices[i]] = f;
                                        ENERGY += f.getMagnitude2();
				}
				#pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[indices[i]] = nCoordinates[indices[i]] + prevCoordinates[indices[i]].getUnitVector() * STEP;
                                }
                        }
			STEP = updateStepLength(STEP, ENERGY, ENERGY0);
                        LOOP++;
                }
                end = omp_get_wtime();
                cout << "Energy:" << ENERGY << endl;
                cout << "Approximation Wall time required:" << end - start << endl;
                result.push_back(ENERGY);
                result.push_back(end - start);
		writeToFile("APPROX"+ to_string(BATCHSIZE)+"PARAOUTRAND" + to_string(ITERATIONS));
                return result;
        }	

	void print(){
		for(INDEXTYPE i = 0; i < graph.rows; i++){
                	cout << "Node:" << i << ", X:" << nCoordinates[i].getX() << ", Y:" << nCoordinates[i].getY()<< endl;
        	}
		cout << endl;
	}
	
	void writeToFile(string f){
		stringstream  data(filename);
    		string lasttok;
    		while(getline(data,lasttok,'/'));
		filename = "datasets/output/" + lasttok + f + ".txt";
		ofstream output;
		output.open(filename);
		cout << "Creating output file in following directory:" << filename << endl;
		for(INDEXTYPE i = 0; i < graph.rows; i++){
			output << nCoordinates[i].getX() <<"\t"<< nCoordinates[i].getY() << "\t" << i+1 << endl;
		}
		output.close();
	}
};
