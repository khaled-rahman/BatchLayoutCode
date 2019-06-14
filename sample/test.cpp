#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include "omp.h"
#include "cstdlib"
using namespace std;

int main(int argc, char *argv[]){
	srand(unsigned(time(0)));
	int t = atoi(argv[1]);
	vector<int> A{0,3,5,7};
	vector<int> B{1,2,3,0,3,0,3,0,1,2};
	vector<int> index;
	for(int i=0; i<5; i++)index.push_back(i);
	for(int i = 0; i<5; i++){
		random_shuffle(index.begin(), index.end());
		for(int j=0; j<index.size(); j++){
			cout << index[j] << " ";
		}
		cout << endl;
	}
	int k = 0, j;
	int v = 9;
	printf("NUmber of threads = %d\n", omp_get_max_threads());
	#pragma omp parallel for num_threads(t)
	for(int i = 0; i < 100; i++){
		int v = 3;
		printf("v = %d, i = %d, threadid = %d\n", v, i, omp_get_thread_num());
		v++;
	}
	printf("v = %d\n", v);
	/*
	for(int i = 0; i<A.size(); i++){
		k = A[i];
		#pragma omp parallel for schedule(static) num_threads(t)
		for(j = 0; j < A.size(); j++){
			if(j == B[k]){
				printf("i(yes):%d # B[%d] = %d, thread = %d\n",i, k, B[k], omp_get_thread_num());
				if(k < B.size() && k < A[i+1]){
					k++;
				}
			}else{
				printf("i(no):%d # j = %d, k = %d, thread =  %d\n",i, j, k, omp_get_thread_num()); 
			}	
		}
	}*/
	/*for(int i = 0; i<A.size(); i++){
                for(j = 0; j < A.size(); j++){
			if(i == j){
				if(i < A.size() - 1){
					for(int k = A[i]; k < A[i+1]; k++)
						printf("Attraction: i = %d --- j = %d\n", i, B[k]);
				}else{
					for(int k = A[i]; k < B.size(); k++)
						printf("Attraction: i = %d --- j = %d\n", i, B[k]);
				}
			}else{
				printf("Repulstion: i = %d --- j = %d\n", i, j);
			}
                }
        }*/
	
	return 0;
}
