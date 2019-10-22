#ifndef _NBLAS_H_
#define _NBLAS_H_

#include "algorithms.h"
#include "newalgo.h"

using namespace std;

#define VALUETYPE double
#define INDEXTYPE int

void NBLAS(INDEXTYPE BATCHSIZE, INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, Coordinate<VALUETYPE> *nCoordinates, VALUETYPE *energy, CSR<INDEXTYPE, VALUETYPE> & graph, VALUETYPE (newalgo::*func)(Coordinate<VALUETYPE> ci, Coordinate<VALUETYPE> cj));

#endif
