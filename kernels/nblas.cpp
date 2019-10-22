#include "nblas.h"
void NBLAS(INDEXTYPE BATCHSIZE, INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, Coordinate<VALUETYPE> *nCoordinates, VALUETYPE *energy, CSR<INDEXTYPE, VALUETYPE> & graph, VALUETYPE (newalgo::*func)(Coordinate<VALUETYPE> ci, Coordinate<VALUETYPE> cj)){

	Coordinate<VALUETYPE>  *prevCoordinates;
	prevCoordinates = static_cast<Coordinate<VALUETYPE> *> (::operator new (sizeof(Coordinate<VALUETYPE>[BATCHSIZE])));
	INDEXTYPE LOOP = 0;
	VALUETYPE STEP = 1.0;	
	VALUETYPE ENERGY = *energy, ENERGY0;
    while(LOOP < ITERATIONS){
	ENERGY0 = ENERGY;
        ENERGY = 0;
	for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1){
		for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                	INDEXTYPE ind = i-b * BATCHSIZE;
                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                       	for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                        	int v = graph.colids[j];
                               	VALUETYPE dx = nCoordinates[v].x - nCoordinates[i].x;
                                VALUETYPE dy = nCoordinates[v].y - nCoordinates[i].y;
                                VALUETYPE d2 = (dx * dx + dy * dy);
                                VALUETYPE di = 1.0 / d2;
                                VALUETYPE d = sqrt(d2);
                               	f = (nCoordinates[v] - nCoordinates[i]);
                                prevCoordinates[ind] += f * (d + di);
                       	}
			for(INDEXTYPE j = 0; j < i; j += 1){
                                //prevCoordinates[ind] += (nCoordinates[j] - nCoordinates[i]) * (this->*func)(nCoordinates[i], nCoordinates[j]);
                        	VALUETYPE dx = nCoordinates[j].x - nCoordinates[i].x;
				VALUETYPE dy = nCoordinates[j].y - nCoordinates[i].y;
				VALUETYPE d = -1.0 / (dx * dx + dy * dy);
				prevCoordinates[ind] += (nCoordinates[j] - nCoordinates[i]) * d;
			}
			for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                //prevCoordinates[ind] += (nCoordinates[j] - nCoordinates[i]) * (this->*func)(nCoordinates[i], nCoordinates[j]);
                        	VALUETYPE dx = nCoordinates[j].x - nCoordinates[i].x;
                                VALUETYPE dy = nCoordinates[j].y - nCoordinates[i].y;
                                VALUETYPE d = -1.0 / (dx * dx + dy * dy);
                                prevCoordinates[ind] += (nCoordinates[j] - nCoordinates[i]) * d;
			}
            	}
		for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                	INDEXTYPE ind = i-b * BATCHSIZE;
                        VALUETYPE d2 = prevCoordinates[ind].x * prevCoordinates[ind].x + prevCoordinates[ind].y * prevCoordinates[ind].y;
                      	VALUETYPE di = 1.0 / sqrt(d2);
                        nCoordinates[i] = nCoordinates[i] + prevCoordinates[ind] * di * STEP;
                        ENERGY += d2;
               	}
	}
	INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
	for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
		Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
		INDEXTYPE ind = i- cleanup;
		for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
			int v = graph.colids[j];
			VALUETYPE dx = nCoordinates[v].x - nCoordinates[i].x;
			VALUETYPE dy = nCoordinates[v].y - nCoordinates[i].y;
			VALUETYPE d2 = (dx * dx + dy * dy);
			VALUETYPE di = 1.0 / d2;
			VALUETYPE d = sqrt(d2);
			f = (nCoordinates[v] - nCoordinates[i]);
			prevCoordinates[ind] += f * (d + di);
		}
		for(INDEXTYPE j = 0; j < i; j += 1){
                        //prevCoordinates[ind] += (nCoordinates[j] - nCoordinates[i]) * (this->*func)(nCoordinates[i], nCoordinates[j]);
			VALUETYPE dx = nCoordinates[j].x - nCoordinates[i].x;
                        VALUETYPE dy = nCoordinates[j].y - nCoordinates[i].y;
                       	VALUETYPE d = -1.0 / (dx * dx + dy * dy);
                        prevCoordinates[ind] += (nCoordinates[j] - nCoordinates[i]) * d;		
		}
		for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                        //prevCoordinates[ind] += (nCoordinates[j] - nCoordinates[i]) * (this->*func)(nCoordinates[i], nCoordinates[j]);
                	VALUETYPE dx = nCoordinates[j].x - nCoordinates[i].x;
                        VALUETYPE dy = nCoordinates[j].y - nCoordinates[i].y;
                        VALUETYPE d = -1.0 / (dx * dx + dy * dy);
                        prevCoordinates[ind] += (nCoordinates[j] - nCoordinates[i]) * d;
		}
	}
	for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
		INDEXTYPE ind = i- cleanup;
		VALUETYPE d2 = prevCoordinates[ind].x * prevCoordinates[ind].x + prevCoordinates[ind].y * prevCoordinates[ind].y;
		VALUETYPE di = 1.0 / sqrt(d2);
		nCoordinates[i] = nCoordinates[i] + prevCoordinates[ind] * di * STEP;
		ENERGY += d2;
	}
	STEP = STEP * 0.999;
        LOOP++;
    }	
}
