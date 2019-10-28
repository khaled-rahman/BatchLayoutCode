#include "newalgo.h"
#include "nblas.h"
#include<immintrin.h>

#define NEARZERO 1e-30

	newalgo::newalgo(CSR<INDEXTYPE, VALUETYPE> &A_csr, string input, string outputd, int init, double weight, double th, string ifile){
		graph.make_empty();
		graph = A_csr;
		outputdir = outputd;
		initfile = ifile;
		//cout << initfile << endl;
		W = weight;
		filename = input;
		threshold = th;
		nCoordinates = static_cast<Coordinate<VALUETYPE> *> (::operator new (sizeof(Coordinate<VALUETYPE>[A_csr.rows])));
		blasX = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[A_csr.rows])));
                blasY = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[A_csr.rows])));
		this->init = init;
	}
	void newalgo::randInit(){
		#pragma omp parallel for schedule(static)
		for(INDEXTYPE i = 0; i < graph.rows; i++){
			VALUETYPE x, y;
			do{
				x = -1.0 + 2.0 * rand()/(RAND_MAX+1.0);
				y = -1.0 + 2.0 * rand()/(RAND_MAX+1.0);
			}while(x * x + y * y > 1.0);
			x = x * MAXMIN;
			y = y * MAXMIN;
			nCoordinates[i] = Coordinate <VALUETYPE>(x, y, i);
		}
	}
	void newalgo::initDFS(){
                int visited[graph.rows] = {0};
                stack <int> STACK;
		double scalefactor = 1.0;
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
                                                nCoordinates[graph.colids[n]] = Coordinate <VALUETYPE>(nCoordinates[parent].getX() + radi*cos(PI*(degree)/180.0), nCoordinates[parent].getY() + radi*sin(PI*(degree)/180.0)) + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
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
                                                nCoordinates[graph.colids[n]] = Coordinate <VALUETYPE>(nCoordinates[parent].getX() + radi*cos(PI*(degree)/180.0), nCoordinates[parent].getY() + radi*sin(PI*(degree)/180.0)) + static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
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
		#pragma omp parallel for schedule(static)
		for(int i = 0; i < graph.rows; i++){
			nCoordinates[i] = nCoordinates[i] * scalefactor;
			blasX[i] = nCoordinates[i].x;
                        blasY[i] = nCoordinates[i].y;
		}
        }

	vector<VALUETYPE> newalgo::batchlayout(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
		INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
		VALUETYPE (newalgo::*frm)(Coordinate<VALUETYPE>, Coordinate<VALUETYPE>);
		frm = &newalgo::frmodel;
                ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                initDFS();
		NBLAS(BATCHSIZE, ITERATIONS, NUMOFTHREADS, nCoordinates, &ENERGY, graph, frm);
                end = omp_get_wtime();
                cout << "NBLAS Minibatch Size:" << BATCHSIZE  << endl;
                cout << "NBLAS Minbatch Energy:" << ENERGY << endl;
                cout << "NBLAS Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFile("NBLAS"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
                result.push_back(ENERGY);
                result.push_back(end - start);
		return result;
	}
	
	vector<VALUETYPE> newalgo::EfficientVersion(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                initDFS();
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
			//#pragma imp parallel for schedule(static)
                        for(int i = 0; i < BATCHSIZE; i++){
                                pb_X[i] = pb_Y[i] = 0;
                        }
		
		//TODO: unrolling and jamming
		// no reverse

                        for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                        VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                        INDEXTYPE ind = i - b * BATCHSIZE;
					for(INDEXTYPE j = 0; j < i; j += 1){
                                                distX = blasX[j] - blasX[i];
                                                distY = blasY[j] - blasY[i];
                                                dist2 = 1.0 / (distX * distX + distY * distY);
                                                fx += distX * dist2;
                                                fy += distY * dist2;
						//if(j == graph.colids[graph.rowptr[i]])
						//printf("Reverse %d: x = %lf, y = %lf\n", j, distX*dist2, distY*dist2);
                                        }
					for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                                distX = blasX[j] - blasX[i];
                                                distY = blasY[j] - blasY[i];
                                                dist2 = 1.0 / (distX * distX + distY * distY);
                                                fx += distX * dist2;
                                                fy += distY * dist2;
						//if(j == graph.colids[graph.rowptr[i]])
                                                //printf("Reverse %d: x = %lf, y = %lf\n", i, distX*dist2, distY*dist2);

                                        }
                                        for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - blasX[i];
                                                distY = blasY[v] - blasY[i];
                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;
                                                pb_X[ind] += distX * dist;
                                                pb_Y[ind] += distY * dist;
						//if(v == graph.colids[graph.rowptr[i]])
						//printf("Forward %d: x = %lf, y = %lf\n", v, distX * dist2, distY*dist2);
                                        }
                                        pb_X[ind] = pb_X[ind] - fx;
                                        pb_Y[ind] = pb_Y[ind] - fy;
                                }
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        INDEXTYPE ind = i - b * BATCHSIZE;
					double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                        ENERGY += dist;
					dist = STEP / sqrt(dist);
					blasX[i] += pb_X[ind] * dist;
                                        blasY[i] += pb_Y[ind] * dist;
					pb_X[ind] = pb_Y[ind] = 0;
                                }
                        }
			//clean up loop
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic)
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
				INDEXTYPE ind = i- cleanup;
                		VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
				for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
					int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = distX * distX + distY * distY;
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;	
				}
				for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                               	for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                               	}
				pb_X[ind] = pb_X[ind] - fx;
				pb_Y[ind] = pb_Y[ind] - fy;
			}
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
				INDEXTYPE ind = i - cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                       	}
                        STEP = STEP * 0.999;
                        LOOP++;
                }
		end = omp_get_wtime();
         #if 0
                cout << "Efficient Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficient Minbatch Energy:" << ENERGY << endl;
                cout << "Efficient Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("Efficient"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

/*
 *    Unrolled version 
 */
#if 0
	vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                initDFS();
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        for(int i = 0; i < BATCHSIZE; i++){
                                pb_X[i] = pb_Y[i] = 0;
                        }
		
			for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1){
				#pragma omp parallel for schedule(static)	
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8){
                                        VALUETYPE fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7;
                                	VALUETYPE fy0, fy1, fy2, fy3, fy4, fy5, fy6, fy7;
                                	VALUETYPE x0, x1, x2, x3, x4, x5, x6, x7;
                                	VALUETYPE y0, y1, y2, y3, y4, y5, y6, y7;
                                	VALUETYPE d0, d1, d2, d3, d4, d5, d6, d7;
					
					int ind = i-b*BATCHSIZE;
					x0 = blasX[i];
					x1 = blasX[i+1];
					x2 = blasX[i+2];
					x3 = blasX[i+3];
					x4 = blasX[i+4];
					x5 = blasX[i+5];
					x6 = blasX[i+6];
					x7 = blasX[i+7];

					y0 = blasY[i];
					y1 = blasY[i+1];	
					y2 = blasY[i+2];
					y3 = blasY[i+3];
					y4 = blasY[i+4];
					y5 = blasY[i+5];
					y6 = blasY[i+6];
					y7 = blasY[i+7];
					
					fx0 = fx1 = fx2 = fx3 = fx4 = fx5 = fx6 = fx7 = 0;
					fy0 = fy1 = fy2 = fy3 = fy4 = fy5 = fy6 = fy7 = 0;		
                                       	VALUETYPE distX, distY, dist; 
					for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x0;
                                                distY = blasY[v] - y0;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx0 += distX * dist;
                                                fy0 += distY * dist;
                                        }
					
					for(INDEXTYPE j = graph.rowptr[i+1]; j < graph.rowptr[i+1+1]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x1;
                                                distY = blasY[v] - y1;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx1 += distX * dist;
                                                fy1 += distY * dist;
                                        }
				
					for(INDEXTYPE j = graph.rowptr[i+2]; j < graph.rowptr[i+1+2]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x2;
                                                distY = blasY[v] - y2;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx2 += distX * dist;
                                                fy2 += distY * dist;
                                        }

					for(INDEXTYPE j = graph.rowptr[i+3]; j < graph.rowptr[i+1+3]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x3;
                                                distY = blasY[v] - y3;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx3 += distX * dist;
                                                fy3 += distY * dist;
                                        }

					for(INDEXTYPE j = graph.rowptr[i+4]; j < graph.rowptr[i+1+4]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x4;
                                                distY = blasY[v] - y4;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx4 += distX * dist;
                                                fy4 += distY * dist;
                                        }
					for(INDEXTYPE j = graph.rowptr[i+5]; j < graph.rowptr[i+1+5]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x5;
                                                distY = blasY[v] - y5;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx5 += distX * dist;
                                                fy5 += distY * dist;
                                        }
					for(INDEXTYPE j = graph.rowptr[i+6]; j < graph.rowptr[i+1+6]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x6;
                                                distY = blasY[v] - y6;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx6 += distX * dist;
                                                fy6 += distY * dist;
                                        }
					for(INDEXTYPE j = graph.rowptr[i+7]; j < graph.rowptr[i+1+7]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - x7;
                                                distY = blasY[v] - y7;

                                                dist = (distX * distX + distY * distY);
                                                dist = sqrt(dist) + 1.0 / dist;


                                                fx7 += distX * dist;
                                                fy7 += distY * dist;
                                        }
					
					for(INDEXTYPE j = 0; j < i; j += 1){
                                        	VALUETYPE vd = blasX[j];
						x0 = vd - x0;
						x1 = vd - x1;
						x2 = vd - x2;
                                                x3 = vd - x3;
						x4 = vd - x4;
                                                x5 = vd - x5;
                                                x6 = vd - x6;
                                                x7 = vd - x7;
						y0 = vd - y0;
						y1 = vd - y1;
						y2 = vd - y2;
                                                y3 = vd - y3;
						y4 = vd - y4;
                                                y5 = vd - y5;
                                                y6 = vd - y6;
                                                y7 = vd - y7; 	       
						//distX = blasX[j] - blasX[i];
                                                //distY = blasY[j] - blasY[i];
                                         	
						d0 = 1.0 / (x0 * x0 + y0 * y0);
						d1 = 1.0 / (x1 * x1 + y1 * y1);
						d2 = 1.0 / (x2 * x2 + y2 * y2);
						d3 = 1.0 / (x3 * x3 + y3 * y3);
						d4 = 1.0 / (x4 * x4 + y4 * y4);
						d5 = 1.0 / (x5 * x5 + y5 * y5);
						d6 = 1.0 / (x6 * x6 + y6 * y6);
						d7 = 1.0 / (x7 * x7 + y7 * y7);       
						//dist2 = 1.0 / (distX * distX + distY * distY);
                                                
						fx0 -= x0 * d0;
						fx1 -= x1 * d1;
						fx2 -= x2 * d2;
                                                fx3 -= x3 * d3;
						fx4 -= x4 * d4;
                                                fx5 -= x5 * d5;
                                                fx6 -= x6 * d6;
                                                fx7 -= x7 * d7;
						fy0 -= y0 * d0;
						fy1 -= y1 * d1;
						fy2 -= y2 * d2;
                                                fy3 -= y3 * d3;
						fy4 -= y4 * d4;
                                                fy5 -= y5 * d5;
                                                fy6 -= y6 * d6;
                                                fy7 -= y7 * d7;
						//fx += distX * dist2;
                                                //fy += distY * dist2;
                                        }
					for(INDEXTYPE j = i+1; j < i + 8; j++){
						x0 = blasX[j] - x0;
						y0 = blasY[j] - y0;
						d0 = 1.0 / (x0 * x0 + y0 * y0);	
						fx0 -= x0 * d0;
						fy0 -= y0 * d0;				
					}
					x1 = blasX[i] - x1;
					y1 = blasX[i] - y1;
					d1 = 1.0 / (x1 * x1 + y1 * y1);
					fx1 -= x1 * d1;
					fy1 -= y1 * d1;
					for(INDEXTYPE j = i+2; j < i + 8; j++){
                                                x1 = blasX[j] - x1;
                                                y1 = blasY[j] - y1;
                                                d1 = 1.0 / (x1 * x1 + y1 * y1);
                                                fx1 -= x1 * d1;
                                                fy1 -= y1 * d1;
                                        }
					for(INDEXTYPE j = i; j < i + 2; j++){
                                                x2 = blasX[j] - x2;
                                                y2 = blasY[j] - y2;
                                                d2 = 1.0 / (x2 * x2 + y2 * y2);
                                                fx2 -= x2 * d2;
                                                fy2 -= y2 * d2;
                                        }
					for(INDEXTYPE j = i+3; j < i + 8; j++){
                                                x2 = blasX[j] - x2;
                                                y2 = blasY[j] - y2;
                                                d2 = d0 = 1.0 / (x2 * x2 + y2 * y2);
                                                fx2 -= x2 * d2;
                                                fy2 -= y2 * d2;
                                        }
					//////////////////
					for(INDEXTYPE j = i; j < i + 3; j++){
                                                x3 = blasX[j] - x3;
                                                y3 = blasY[j] - y3;
                                                d3 = 1.0 / (x3 * x3 + y3 * y3);
                                                fx3 -= x3 * d3;
                                                fy3 -= y3 * d3;
                                        }
					for(INDEXTYPE j = i+4; j < i + 8; j++){
                                                x3 = blasX[j] - x3;
                                                y3 = blasY[j] - y3;
                                                d3 = 1.0 / (x3 * x3 + y3 * y3);
                                                fx3 -= x3 * d3;
                                                fy3 -= y3 * d3;
                                        }
					////////
					for(INDEXTYPE j = i; j < i + 4; j++){
                                                x4 = blasX[j] - x4;
                                                y4 = blasY[j] - y4;
                                                d4 = 1.0 / (x4 * x4 + y4 * y4);
                                                fx4 -= x4 * d4;
                                                fy4 -= y4 * d4;
                                        }
					for(INDEXTYPE j = i+5; j < i + 8; j++){
                                                x4 = blasX[j] - x4;
                                                y4 = blasY[j] - y4;
                                                d4 = 1.0 / (x4 * x4 + y4 * y4);
                                                fx4 -= x4 * d4;
                                                fy4 -= y4 * d4;
                                        }
					/////
					for(INDEXTYPE j = i; j < i + 5; j++){
                                                x5 = blasX[j] - x5;
                                                y5 = blasY[j] - y5;
                                                d5 = 1.0 / (x5 * x5 + y5 * y5);
                                                fx5 -= x5 * d5;
                                                fy5 -= y5 * d5;
                                        }
					for(INDEXTYPE j = i+6; j < i + 8; j++){
                                                x5 = blasX[j] - x5;
                                                y5 = blasY[j] - y5;
                                                d5 = 1.0 / (x5 * x5 + y5 * y5);
                                                fx5 -= x5 * d5;
                                                fy5 -= y5 * d5;
                                        }
			
					//////
					for(INDEXTYPE j = i; j < i + 6; j++){
                                                x6 = blasX[j] - x6;
                                                y6 = blasY[j] - y6;
                                                d6 = 1.0 / (x6 * x6 + y6 * y6);
                                                fx6 -= x6 * d6;
                                                fy6 -= y6 * d6;
                                        }
					x6 = blasX[i+7] - x6;
                                        y6 = blasX[i+7] - y6;
                                        d6 = 1.0 / (x6 * x6 + y6 * y6);
                                        fx6 -= x6 * d6;
                                        fy6 -= y6 * d1;
					////
					for(INDEXTYPE j = i; j < i + 7; j++){
                                                x7 = blasX[j] - x7;
                                                y7 = blasY[j] - y7;
                                                d7 = 1.0 / (x7 * x7 + y7 * y7);
                                                fx7 -= x7 * d7;
                                                fy7 -= y7 * d7;
                                        }
					////////////
                                        for(INDEXTYPE j = i+8; j < graph.rows; j += 1){
                                                VALUETYPE vx = blasX[j];
						VALUETYPE vy = blasY[j];
						x0 = vx - x0;
                                                x1 = vx - x1;
                                                x2 = vx - x2;
                                                x3 = vx - x3;
                                                x4 = vx - x4;
                                                x5 = vx - x5;
                                                x6 = vx - x6;
                                                x7 = vx - x7;
                                                y0 = vy - y0;
                                                y1 = vy - y1;
                                                y2 = vy - y2;
                                                y3 = vy - y3;
                                                y4 = vy - y4;
                                                y5 = vy - y5;
                                                y6 = vy - y6;
                                                y7 = vy - y7;
						//distX = blasX[j] - blasX[i];
                                                //distY = blasY[j] - blasY[i];
                                                
						d0 = 1.0 / (x0 * x0 + y0 * y0);
                                                d1 = 1.0 / (x1 * x1 + y1 * y1);
                                                d2 = 1.0 / (x2 * x2 + y2 * y2);
                                                d3 = 1.0 / (x3 * x3 + y3 * y3);
                                                d4 = 1.0 / (x4 * x4 + y4 * y4);
                                                d5 = 1.0 / (x5 * x5 + y5 * y5);
                                                d6 = 1.0 / (x6 * x6 + y6 * y6);
                                                d7 = 1.0 / (x7 * x7 + y7 * y7);
						//dist2 = 1.0 / (distX * distX + distY * distY);
                                                
						fx0 -= x0 * d0;
                                                fx1 -= x1 * d1;
                                                fx2 -= x2 * d2;
                                                fx3 -= x3 * d3;
                                                fx4 -= x4 * d4;
                                                fx5 -= x5 * d5;
                                                fx6 -= x6 * d6;
                                                fx7 -= x7 * d7;
                                                fy0 -= y0 * d0;
                                                fy1 -= y1 * d1;
                                                fy2 -= y2 * d2;
                                                fy3 -= y3 * d3;
                                                fy4 -= y4 * d4;
                                                fy5 -= y5 * d5;
                                                fy6 -= y6 * d6;
                                                fy7 -= y7 * d7;
						//fx += distX * dist2;
                                                //fy += distY * dist2;
                                        }
					pb_X[ind] = fx0;
					pb_X[ind+1] = fx1;
					pb_X[ind+2] = fx2;
                                        pb_X[ind+3] = fx3;
					pb_X[ind+4] = fx4;
                                        pb_X[ind+5] = fx5;
                                        pb_X[ind+6] = fx6;
                                        pb_X[ind+7] = fx7;
					
					pb_Y[ind] = fy0;
					pb_Y[ind+1] = fy1;
					pb_Y[ind+2] = fy2;
                                        pb_Y[ind+3] = fy3;
					pb_Y[ind+4] = fy4;
                                        pb_Y[ind+5] = fy5;
                                        pb_Y[ind+6] = fy6;
                                        pb_Y[ind+7] = fy7;	
				}
				/*printf("After:\n");
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
                                }
				*/
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                        int ind = i-b*BATCHSIZE;
                                        VALUETYPE d;
					d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
					ENERGY += d ;
					d = STEP / sqrt(d);
					blasX[i] += pb_X[ind] * d;
                                        blasY[i] += pb_Y[ind] * d;
					pb_X[ind] = 0;
					pb_Y[ind] = 0;    
                                }
                        }
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
            #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
            #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
#else 
/*
 * ************************ N-DIM : Unroll and Jam Impl ******************************
 * NOTE: define for appropriate unroll factor 
 *
 * NOTE: NOTE: 
 *             Unroll = 8 .... vectorized by compiler... 
 *             So, implicit unrolling = 8 * 8 = 64
 *             So, N-dim threading .... only 256 / 64 = 4 threads is active!!!
 * When Unroll = 1  // compiler vectorized the code so, implicit unrolling 8
 *    Active iteration = 256 / 8 = 32 
 *    18 threads, per share 1+ 
 *    NOTE: that's why when Unroll = 2, got less than 1 iteration !!!!! 
 */
   //#define UR4_MEMOPT 1 
   #define UR8 1 
   #define DEBUG 0

   #ifdef UR8 
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as long as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7;
               VALUETYPE fy0, fy1, fy2, fy3, fy4, fy5, fy6, fy7;
               
               
               VALUETYPE x0, x1, x2, x3, x4, x5, x6, x7;
               VALUETYPE y0, y1, y2, y3, y4, y5, y6, y7;
               VALUETYPE d0, d1, d2, d3, d4, d5, d6, d7;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];
	       x4 = blasX[i+4];
	       x5 = blasX[i+5];
	       x6 = blasX[i+6];
	       x7 = blasX[i+7];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
	       y4 = blasY[i+4];
	       y5 = blasY[i+5];
	       y6 = blasY[i+6];
	       y7 = blasY[i+7];
					
	       fx0 = fx1 = fx2 = fx3 = fx4 = fx5 = fx6 = fx7 = 0;
	       fy0 = fy1 = fy2 = fy3 = fy4 = fy5 = fy6 = fy7 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7;
                  VALUETYPE dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
		  dx2 = xj - x2;
                  dx3 = xj - x3;
		  dx4 = xj - x4;
                  dx5 = xj - x5;
                  dx6 = xj - x6;
                  dx7 = xj - x7;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		  dy2 = yj - y2;
                  dy3 = yj - y3;
		  dy4 = yj - y4;
                  dy5 = yj - y5;
                  dy6 = yj - y6;
                  dy7 = yj - y7; 	       
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		  d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		  d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		  d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		  d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		  d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		  d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		  fx2 += dx2 * d2;
                  fx3 += dx3 * d3;
		  fx4 += dx4 * d4;
                  fx5 += dx5 * d5;
                  fx6 += dx6 * d6;
                  fx7 += dx7 * d7;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		  fy2 += dy2 * d2;
                  fy3 += dy3 * d3;
		  fy4 += dy4 * d4;
                  fy5 += dy5 * d5;
                  fy6 += dy6 * d6;
                  fy7 += dy7 * d7;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
               for (INDEXTYPE j = i; j < i+8; j++) 
               {
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  if (j != i)
                  {
                     VALUETYPE dx0, dy0;
                     dx0 = xj - x0;
                     dy0 = yj - y0;
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     fx0 += dx0 * d0;
                     fy0 += dy0 * d0;
                  }
                  if ( j != i+1 )
                  {
                     VALUETYPE dx1, dy1;
                     dx1 = xj - x1;
                     dy1 = yj - y1;
	             d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     fx1 += dx1 * d1;
                     fy1 += dy1 * d1;
                  }
                  if ( j != i+2 )
                  {
                     VALUETYPE dx2, dy2;
                     dx2 = xj - x2;
                     dy2 = yj - y2;
	             d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     fx2 += dx2 * d2;
                     fy2 += dy2 * d2;
                  }
                  if ( j != i+3 )
                  {
                     VALUETYPE dx3, dy3;
                     dx3 = xj - x3;
                     dy3 = yj - y3;
	             d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     fx3 += dx3 * d3;
                     fy3 += dy3 * d3;
                  }
                  if ( j != i+4 )
                  {
                     VALUETYPE dx4, dy4;
                     dx4 = xj - x4;
                     dy4 = yj - y4;
	             d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     fx4 += dx4 * d4;
                     fy4 += dy4 * d4;
                  }
                  if ( j != i+5 )
                  {
                     VALUETYPE dx5, dy5;
                     dx5 = xj - x5;
                     dy5 = yj - y5;
	             d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     fx5 += dx5 * d5;
                     fy5 += dy5 * d5;
                  }
                  if ( j != i+6 )
                  {
                     VALUETYPE dx6, dy6;
                     dx6 = xj - x6;
                     dy6 = yj - y6;
	             d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     fx6 += dx6 * d6;
                     fy6 += dy6 * d6;
                  }
                  if ( j != i+7 )
                  {
                     VALUETYPE dx7, dy7;
                     dx7 = xj - x7;
                     dy7 = yj - y7;
	             d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);
		     fx7 += dx7 * d7;
                     fy7 += dy7 * d7;
                  }
               }
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7;
                  VALUETYPE dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
		  dx2 = xj - x2;
                  dx3 = xj - x3;
		  dx4 = xj - x4;
                  dx5 = xj - x5;
                  dx6 = xj - x6;
                  dx7 = xj - x7;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		  dy2 = yj - y2;
                  dy3 = yj - y3;
		  dy4 = yj - y4;
                  dy5 = yj - y5;
                  dy6 = yj - y6;
                  dy7 = yj - y7; 	       
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		  d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		  d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		  d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		  d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		  d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		  d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		  fx2 += dx2 * d2;
                  fx3 += dx3 * d3;
		  fx4 += dx4 * d4;
                  fx5 += dx5 * d5;
                  fx6 += dx6 * d6;
                  fx7 += dx7 * d7;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		  fy2 += dy2 * d2;
                  fy3 += dy3 * d3;
		  fy4 += dy4 * d4;
                  fy5 += dy5 * d5;
                  fy6 += dy6 * d6;
                  fy7 += dy7 * d7;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
            
            #if DEBUG == 1
                  cout << "Loop = "<< LOOP <<" i = " << i << 
                     ": connected nodes = " << 
                     graph.rowptr[i+1] - graph.rowptr[i] << endl;
            #endif
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
            #if DEBUG == 1
                  cout << "i = " << i+1 << ": connected nodes = " << 
                     graph.rowptr[i+2] - graph.rowptr[i+1] << endl;
            #endif
					
	       for(INDEXTYPE j = graph.rowptr[i+1]; j < graph.rowptr[i+1+1]; j += 1)
               {
                  int v = graph.colids[j];
                  
                  distX = blasX[v] - x1;
                  distY = blasY[v] - y1;
 
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px1 += distX * dist;
                  py1 += distY * dist;
               }
				
            #if DEBUG == 1
                  cout << "i = " << i+2 << ": connected nodes = " << 
                     graph.rowptr[i+3] - graph.rowptr[i+2] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+2]; j < graph.rowptr[i+1+2]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x2;
                  distY = blasY[v] - y2;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px2 += distX * dist;
                  py2 += distY * dist;
               }

            #if DEBUG == 1
                  cout << "i = " << i+3 << ": connected nodes = " << 
                     graph.rowptr[i+4] - graph.rowptr[i+3] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+3]; j < graph.rowptr[i+1+3]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x3;
                  distY = blasY[v] - y3;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px3 += distX * dist;
                  py3 += distY * dist;                         
               }

            #if DEBUG == 1
                  cout << "i = " << i+4 << ": connected nodes = " << 
                     graph.rowptr[i+5] - graph.rowptr[i+4] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+4]; j < graph.rowptr[i+1+4]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x4;
                  distY = blasY[v] - y4;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px4 += distX * dist;
                  py4 += distY * dist;
               }
            #if DEBUG == 1
                  cout << "i = " << i+5 << ": connected nodes = " << 
                     graph.rowptr[i+6] - graph.rowptr[i+5] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+5]; j < graph.rowptr[i+1+5]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x5;
                  distY = blasY[v] - y5;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px5 += distX * dist;
                  py5 += distY * dist;
               }
            #if DEBUG == 1
                  cout << "i = " << i+6 << ": connected nodes = " << 
                     graph.rowptr[i+7] - graph.rowptr[i+6] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+6]; j < graph.rowptr[i+1+6]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x6;
                  distY = blasY[v] - y6;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px6 += distX * dist;
                  py6 += distY * dist;                         
               }
            #if DEBUG == 1
                  cout << "i = " << i+7 << ": connected nodes = " << 
                     graph.rowptr[i+8] - graph.rowptr[i+7] << endl;
            #endif
	       for(INDEXTYPE j = graph.rowptr[i+7]; j < graph.rowptr[i+1+7]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x7;
                  distY = blasY[v] - y7;
   
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px7 += distX * dist;
                  py7 += distY * dist;
                }
	       
               pb_X[ind] = px0 - fx0;
	       pb_X[ind+1] = px1 - fx1;
	       pb_X[ind+2] = px2 - fx2;
               pb_X[ind+3] = px3 - fx3;
	       pb_X[ind+4] = px4 - fx4;
               pb_X[ind+5] = px5 - fx5;
               pb_X[ind+6] = px6 - fx6;
               pb_X[ind+7] = px7- fx7;
					
	       pb_Y[ind] = py0 - fy0;
	       pb_Y[ind+1] = py1 - fy1;
	       pb_Y[ind+2] = py2 - fy2;
               pb_Y[ind+3] = py3 - fy3;
	       pb_Y[ind+4] = py4 - fy4;
               pb_Y[ind+5] = py5 - fy5;
               pb_Y[ind+6] = py5 - fy6;
               pb_Y[ind+7] = py6 - fy7;	
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
            #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
            #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
   #elif defined(UR4)
/*
 * **************************** UNROLL FACTOR = 4 ************************ 
 */
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 4)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0, fx1, fx2, fx3;
               VALUETYPE fy0, fy1, fy2, fy3;
               
               
               VALUETYPE x0, x1, x2, x3;
               VALUETYPE y0, y1, y2, y3;
               VALUETYPE d0, d1, d2, d3;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
					
	       fx0 = fx1 = fx2 = fx3 = 0;
	       fy0 = fy1 = fy2 = fy3 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1, dx2, dx3;
                  VALUETYPE dy0, dy1, dy2, dy3;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
		  dx2 = xj - x2;
                  dx3 = xj - x3;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		  dy2 = yj - y2;
                  dy3 = yj - y3;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		  d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		  d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		  fx2 += dx2 * d2;
                  fx3 += dx3 * d3;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		  fy2 += dy2 * d2;
                  fy3 += dy3 * d3;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
               for (INDEXTYPE j = i; j < i+4; j++) 
               {
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  if (j != i)
                  {
                     VALUETYPE dx0, dy0;
                     dx0 = xj - x0;
                     dy0 = yj - y0;
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     fx0 += dx0 * d0;
                     fy0 += dy0 * d0;
                  }
                  if ( j != i+1 )
                  {
                     VALUETYPE dx1, dy1;
                     dx1 = xj - x1;
                     dy1 = yj - y1;
	             d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     fx1 += dx1 * d1;
                     fy1 += dy1 * d1;
                  }
                  if ( j != i+2 )
                  {
                     VALUETYPE dx2, dy2;
                     dx2 = xj - x2;
                     dy2 = yj - y2;
	             d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     fx2 += dx2 * d2;
                     fy2 += dy2 * d2;
                  }
                  if ( j != i+3 )
                  {
                     VALUETYPE dx3, dy3;
                     dx3 = xj - x3;
                     dy3 = yj - y3;
	             d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     fx3 += dx3 * d3;
                     fy3 += dy3 * d3;
                  }
               }
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+4; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1, dx2, dx3;
                  VALUETYPE dy0, dy1, dy2, dy3;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
		  dx2 = xj - x2;
                  dx3 = xj - x3;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		  dy2 = yj - y2;
                  dy3 = yj - y3;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		  d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		  d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		  fx2 += dx2 * d2;
                  fx3 += dx3 * d3;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		  fy2 += dy2 * d2;
                  fy3 += dy3 * d3;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
					
	       for(INDEXTYPE j = graph.rowptr[i+1]; j < graph.rowptr[i+1+1]; j += 1)
               {
                  int v = graph.colids[j];
                  
                  distX = blasX[v] - x1;
                  distY = blasY[v] - y1;
 
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px1 += distX * dist;
                  py1 += distY * dist;
               }
				
	       for(INDEXTYPE j = graph.rowptr[i+2]; j < graph.rowptr[i+1+2]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x2;
                  distY = blasY[v] - y2;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px2 += distX * dist;
                  py2 += distY * dist;
               }

	       for(INDEXTYPE j = graph.rowptr[i+3]; j < graph.rowptr[i+1+3]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x3;
                  distY = blasY[v] - y3;

                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px3 += distX * dist;
                  py3 += distY * dist;                         
               }

               pb_X[ind] = px0 - fx0;
	       pb_X[ind+1] = px1 - fx1;
	       pb_X[ind+2] = px2 - fx2;
               pb_X[ind+3] = px3 - fx3;
					
	       pb_Y[ind] = py0 - fy0;
	       pb_Y[ind+1] = py1 - fy1;
	       pb_Y[ind+2] = py2 - fy2;
               pb_Y[ind+3] = py3 - fy3;
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
               writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
   #elif defined (UR2)
/*
 * **************************** UNROLL FACTOR = 2 ************************ 
 */
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if DEGUG == 1
      cout << "*************** Unroll factor = 2" << endl;
      cout << "Rows = " << graph.rows << endl;
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 2)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0, fx1;
               VALUETYPE fy0, fy1;
               
               
               VALUETYPE x0, x1;
               VALUETYPE y0, y1;
               VALUETYPE d0, d1;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
					
	       fx0 = fx1 = 0;
	       fy0 = fy1 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
               for (INDEXTYPE j = i; j < i+2; j++) 
               {
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  if (j != i)
                  {
                     VALUETYPE dx0, dy0;
                     dx0 = xj - x0;
                     dy0 = yj - y0;
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     fx0 += dx0 * d0;
                     fy0 += dy0 * d0;
                  }
                  if ( j != i+1 )
                  {
                     VALUETYPE dx1, dy1;
                     dx1 = xj - x1;
                     dy1 = yj - y1;
	             d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     fx1 += dx1 * d1;
                     fy1 += dy1 * d1;
                  }
               }
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+2; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
		  dx1 = xj - x1;
	          
                  dy0 = yj - y0;
		  dy1 = yj - y1;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		  d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		  fx1 += dx1 * d1;
		
                  fy0 += dy0 * d0;
		  fy1 += dy1 * d1;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
					
	       for(INDEXTYPE j = graph.rowptr[i+1]; j < graph.rowptr[i+1+1]; j += 1)
               {
                  int v = graph.colids[j];
                  
                  distX = blasX[v] - x1;
                  distY = blasY[v] - y1;
 
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px1 += distX * dist;
                  py1 += distY * dist;
               }
				
               pb_X[ind] = px0 - fx0;
	       pb_X[ind+1] = px1 - fx1;
					
	       pb_Y[ind] = py0 - fy0;
	       pb_Y[ind+1] = py1 - fy1;
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
            #if DEBUG == 1
                        cout << "Total cleanup nodes = " << graph.rows - cleanup;
            #endif
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
   #elif defined(UR1)
/*
 * **************************** UNROLL FACTOR = 1 ************************ 
 */
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0;
               VALUETYPE fy0;
               
               
               VALUETYPE x0;
               VALUETYPE y0;
               VALUETYPE d0;
					
	       x0 = blasX[i];

	       y0 = blasY[i];
					
	       fx0 = 0;
	       fy0 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
                  dy0 = yj - y0;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
                  fy0 += dy0 * d0;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+1; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
                  dy0 = yj - y0;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		
                  fy0 += dy0 * d0;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
				
               pb_X[ind] = px0 - fx0;
	       pb_Y[ind] = py0 - fy0;
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
   #elif defined(UR4_MEMOPT)
/*
 * **************************** MEMOPT: UNROLL FACTOR = 4 **************** 
 */
   vector<VALUETYPE> newalgo::EfficientVersionUnRoll(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor
 *          NOTE: calling omp parallel for has overhead for fork joining 
 *          multiple times... so, handle loop manually 
 *
 *          implicit unrolling = 4 * 8 = 32 .. 
 *          per thread slice = (256 / 32) / 18 = .... doesn't make sense???
 */
            #pragma omp parallel 
            {
               int id, nthreads, chunksize;

               id = omp_get_thread_num();
               nthreads = omp_get_num_threads();
               chunksize = BATCHSIZE / nthreads; 

	       for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 4)
               {
	          int ind = i-b*BATCHSIZE;

                  VALUETYPE distX, distY, dist; 
                  VALUETYPE fx0, fx1, fx2, fx3;
                  VALUETYPE fy0, fy1, fy2, fy3;
                  VALUETYPE x0, x1, x2, x3;
                  VALUETYPE y0, y1, y2, y3;
                  VALUETYPE d0, d1, d2, d3;
					
	          x0 = blasX[i];
	          x1 = blasX[i+1];
	          x2 = blasX[i+2];
	          x3 = blasX[i+3];

	          y0 = blasY[i];
	          y1 = blasY[i+1];	
	          y2 = blasY[i+2];
	          y3 = blasY[i+3];
					
	          fx0 = fx1 = fx2 = fx3 = 0;
	          fy0 = fy1 = fy2 = fy3 = 0;		
/*
 *              j is up to i... lower up case  
 */
                  for(INDEXTYPE j = 0; j < i; j += 1)
                  {
                     VALUETYPE dx0, dx1, dx2, dx3;
                     VALUETYPE dy0, dy1, dy2, dy3;
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		     fx0 += dx0 * d0;
		     fx1 += dx1 * d1;
		     fx2 += dx2 * d2;
                     fx3 += dx3 * d3;
		
                     fy0 += dy0 * d0;
		     fy1 += dy1 * d1;
		     fy2 += dy2 * d2;
                     fy3 += dy3 * d3;
                  }		
/*
 *                location where we need to skip some points i==j
 *                NOTE: its UR iteration, no need to optimize 
 *                Hopefully compiler will unroll and get rid of the conditions
 *                NOTE: can be optimized it using K-reg with intrinsic
 */
                  for (INDEXTYPE j = i; j < i+4; j++) 
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     if (j != i)
                     {
                        VALUETYPE dx0, dy0;
                        dx0 = xj - x0;
                        dy0 = yj - y0;
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        fx0 += dx0 * d0;
                        fy0 += dy0 * d0;
                     }
                     if ( j != i+1 )
                     {
                        VALUETYPE dx1, dy1;
                        dx1 = xj - x1;
                        dy1 = yj - y1;
	                d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        fx1 += dx1 * d1;
                        fy1 += dy1 * d1;
                     }
                     if ( j != i+2 )
                     {
                        VALUETYPE dx2, dy2;
                        dx2 = xj - x2;
                        dy2 = yj - y2;
	                d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        fx2 += dx2 * d2;
                        fy2 += dy2 * d2;
                     }
                     if ( j != i+3 )
                     {
                        VALUETYPE dx3, dy3;
                        dx3 = xj - x3;
                        dy3 = yj - y3;
	                d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        fx3 += dx3 * d3;
                        fy3 += dy3 * d3;
                     }
                  }
/*
 *                Upper part  
 */ 
                  for(INDEXTYPE j = i+4; j < graph.rows; j += 1)
                  {
                     VALUETYPE dx0, dx1, dx2, dx3;
                     VALUETYPE dy0, dy1, dy2, dy3;
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		     fx0 += dx0 * d0;
		     fx1 += dx1 * d1;
		     fx2 += dx2 * d2;
                     fx3 += dx3 * d3;
		
                     fy0 += dy0 * d0;
		     fy1 += dy1 * d1;
		     fy2 += dy2 * d2;
                     fy3 += dy3 * d3;
                  }
                  pb_X[ind] = fx0;
	          pb_X[ind+1] = fx1;
	          pb_X[ind+2] = fx2;
                  pb_X[ind+3] = fx3;
					
	          pb_Y[ind] = fy0;
	          pb_Y[ind+1] = fy1;
	          pb_Y[ind+2] = fy2;
                  pb_Y[ind+3] = fy3;
	       }
/*
 *          Remove connected node calc out of the nested loop
 *          FIXME: ***** the code slows down by 10% 
 *
 */
#if 0
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }

               pbX = pbX - pb_X[ind];
               pbY = pbY - pb_Y[ind];
	       
               dist = (pbX * pbX + pbY * pbY);
               #pragma omp atomic
               ENERGY += dist ;
	       
               dist = STEP / sqrt(dist);
	       blasX[i] += pbX * dist;
               blasY[i] += pbY * dist;
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
#else
/*
 *          Calc connected nodes 
 *          complexity : BATCHSIZE * avg connection 
 *          ************ still 10% slow, why ????? 
 *          Unrolling by 4, we eliminate 4 * rows memory loads, 
 *          but moving out the connected nodes, we have extra loads 
 *          = BATCHSIZE * avg connection
 *          ***** Need to find:  4*VLEN*rows >> BATCHSIZE * avg connection
 *                         example: pkustk01
 *                                  avg connect = 50 (guessing from prof)
 *                               4*8*22044  >> 256 * 50
 *                               705408     >> 12800 
 *
 */
	       //#pragma omp parallel for schedule(static)	
	       for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
               {
                  int ind = i-b*BATCHSIZE;
                  VALUETYPE pbX=0.0, pbY=0.0;
                  VALUETYPE dist, distX, distY;
                  for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
                  {
                     int v = graph.colids[j];
                     distX = blasX[v] - blasX[i];
                     distY = blasY[v] - blasY[i];
                     dist = (distX * distX + distY * distY);
                     dist = sqrt(dist) + 1.0 / dist;
                     pbX += distX * dist;
                     pbY += distY * dist;
                  }
                  pb_X[ind] = pbX - pb_X[ind];
                  pb_Y[ind] = pbY - pb_Y[ind];
               }
            }
/*
 *          Update nodes 
 */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
#endif
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
               writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

   #endif

#endif



	vector<VALUETYPE> newalgo::EfficientVersionV2(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                initDFS();
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        for(int i = 0; i < BATCHSIZE; i++){
                                pb_X[i] = pb_Y[i] = 0;
                        }
			for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1){
                                //#pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8){
                                        VALUETYPE x, y, fx = 0, fy = 0, distX, distY, dist, dist2;
					register __m512d vbx0, vbx1, vbx2, vbx3, vbx4, vbx5, vbx6, vbx7, dc;
					register __m512d vby0, vby1, vby2, vby3, vby4, vby5, vby6, vby7;
					register __m512d vfx0, vfx1, vfx2, vfx3, vfx4, vfx5, vfx6, vfx7;
                                        register __m512d vfy0, vfy1, vfy2, vfy3, vfy4, vfy5, vfy6, vfy7;
					
					vfx0 = _mm512_set1_pd(0.0);
					vfy0 = _mm512_set1_pd(0.0);
					dc = _mm512_set1_pd(1.0);

					vbx0 = _mm512_loadu_pd(blasX + i);
					vby0 = _mm512_loadu_pd(blasY + i);
                                        
					int ind = i-b*BATCHSIZE;
                                        for(INDEXTYPE j = 0; j < i; j += 1){
                                                register __m512d vcx, vcy, d0, d1, d2, d3, d4, d5, d6, d7;
							
						vcx = _mm512_set1_pd(blasX[j]);
						vcy = _mm512_set1_pd(blasY[j]);
						
						vbx0 = _mm512_sub_pd(vcx, vbx0);
						vby0 = _mm512_sub_pd(vcy, vby0);
						//distX = blasX[j] - blasX[i];
                                                //distY = blasY[j] - blasY[i];
                                                
						d0 = _mm512_mul_pd(vbx0, vbx0);
						d0 = _mm512_fmadd_pd(vby0, vby0, d0);
						d0 = _mm512_div_pd(dc, d0);
						//dist2 = 1.0 / (distX * distX + distY * distY);
                                                
						vfx0 = _mm512_fmadd_pd(vbx0, d0, vfx0);
						vfy0 = _mm512_fmadd_pd(vby0, d0, vfy0);
						//fx += distX * dist2;
                                                //fy += distY * dist2;
                                        }
                                        for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                                register __m512d vcx, vcy, d0, d1, d2, d3, d4, d5, d6, d7;

                                                vcx = _mm512_set1_pd(blasX[j]);
                                                vcy = _mm512_set1_pd(blasY[j]);

                                                vbx0 = _mm512_sub_pd(vcx, vbx0);
                                                vby0 = _mm512_sub_pd(vcy, vby0);
						//distX = blasX[j] - blasX[i];
                                                //distY = blasY[j] - blasY[i];
                                                
						d0 = _mm512_mul_pd(vbx0, vbx0);
                                                d0 = _mm512_fmadd_pd(vby0, vby0, d0);
                                                d0 = _mm512_div_pd(dc, d0);
						//dist2 = 1.0 / (distX * distX + distY * distY);
                                                
						vfx0 = _mm512_fmadd_pd(vbx0, d0, vfx0);
                                                vfy0 = _mm512_fmadd_pd(vby0, d0, vfy0);
						//fx += distX * dist2;
                                                //fy += distY * dist2;
                                        }
					_mm512_storeu_pd(pb_X+ind, vfx0);
					_mm512_storeu_pd(pb_Y+ind, vfy0);
					for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                                int v = graph.colids[j];
                                                distX = blasX[v] - blasX[i];
                                                distY = blasY[v] - blasY[i];
                                              	dist2 = (distX * distX + distY * distY);
						dist = sqrt(dist2) + 1.0 / (dist2);
						pb_X[ind] += distX * dist;
                                                pb_Y[ind] += distY * dist;
                                        }
                                        //pb_X[ind] = pb_X[ind] - fx;
                                        //pb_Y[ind] = pb_Y[ind] - fy;
                                }
				printf("After:\n");
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
					printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
				}
				printf("\n");
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        int ind = i-b*BATCHSIZE;
					double dist2 = pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind];
                                        double dist = 1.0 / sqrt(dist2);
                                        blasX[i] += pb_X[ind] * STEP * dist;
                                        blasY[i] += pb_Y[ind] * STEP * dist;
                                        ENERGY += (dist2);
                                }
                        }
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
                        //#pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist2 = distX * distX + distY * distY;
                                        dist = sqrt(dist2) + 1.0 / dist2;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (1.0 * STEP) / sqrt(pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
                                ENERGY += (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
                cout << "Efficient V2 Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficient V2 Minbatch Energy:" << ENERGY << endl;
                cout << "Efficient V2 Minibatch Parallel Wall time required:" << end - start << endl;
                //writeToFile("EfficientV2"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

	void newalgo::print(){
		for(INDEXTYPE i = 0; i < graph.rows; i++){
                	cout << "Node:" << i << ", X:" << nCoordinates[i].getX() << ", Y:" << nCoordinates[i].getY()<< endl;
        	}
		cout << endl;
	}
	void newalgo::writeToFile(string f){
		stringstream  data(filename);
    		string lasttok;
    		while(getline(data,lasttok,'/'));
		filename = outputdir + lasttok + f + ".txt";
		ofstream output;
		output.open(filename);
		cout << "Creating output file in following directory:" << filename << endl;
		for(INDEXTYPE i = 0; i < graph.rows; i++){
			output << nCoordinates[i].getX() <<"\t"<< nCoordinates[i].getY() << "\t" << i+1 << endl;
		}
		output.close();
	}
	void newalgo::writeToFileEFF(string f){
		stringstream  data(filename);
                string lasttok;
                while(getline(data,lasttok,'/'));
                filename = outputdir + lasttok + f + ".txt";
                ofstream output;
                output.open(filename);
                cout << "Creating output file in following directory:" << filename << endl;
                for(INDEXTYPE i = 0; i < graph.rows; i++){
                        output << blasX[i] <<"\t"<< blasY[i] << "\t" << i+1 << endl;
                }
                output.close();
	}
/*
 *       Using intrinsic codes .... 
 *    
 */
/*
 * **************************** Intrinsic Version ************************ 
 */
#if 0
   vector<VALUETYPE> newalgo::EfficientVersionVec(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {

/*
 *          NOTE: no cleanup for unroll as lon as BATCHZIE is multiple of unroll
 *          factor 
 */
	    #pragma omp parallel for schedule(static)	
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8)
            {
	       int ind = i-b*BATCHSIZE;

	       register __m512d vbx0, vbx1, vbx2, vbx3, vbx4, vbx5, vbx6, vbx7, dc;
               
               VALUETYPE distX, distY, dist; 
               VALUETYPE fx0;
               VALUETYPE fy0;
               
               
               VALUETYPE x0;
               VALUETYPE y0;
               VALUETYPE d0;
					
	       x0 = blasX[i];

	       y0 = blasY[i];
					
	       fx0 = 0;
	       fy0 = 0;		
		
/*
 *              j is up to i... lower up case  
 */
               for(INDEXTYPE j = 0; j < i; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
                  dy0 = yj - y0;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
                  fy0 += dy0 * d0;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }		
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
/*
 *             Upper part  
 *
 */ 
               for(INDEXTYPE j = i+1; j < graph.rows; j += 1)
               {
                  VALUETYPE dx0, dx1;
                  VALUETYPE dy0, dy1;
                  VALUETYPE xj = blasX[j];
                  VALUETYPE yj = blasY[j];
		  
                  dx0 = xj - x0;
                  dy0 = yj - y0;
		
                  //distX = blasX[j] - blasX[i];
                  //distY = blasY[j] - blasY[i];
                                   	
	          d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		
                  //dist2 = 1.0 / (distX * distX + distY * distY);
                                            
		  fx0 += dx0 * d0;
		
                  fy0 += dy0 * d0;
		 
                  //fx += distX * dist2;
                  //fy += distY * dist2;
               }
               
               
               // connected nodes 
               VALUETYPE px0=0, px1=0, px2=0, px3=0, px4=0, px5=0, px6=0, px7=0;
               VALUETYPE py0=0, py1=0, py2=0, py3=0, py4=0, py5=0, py6=0, py7=0;
               
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - x0;
                  distY = blasY[v] - y0;
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;

                  px0 += distX * dist;
                  py0 += distY * dist;
               }
				
               pb_X[ind] = px0 - fx0;
	       pb_Y[ind] = py0 - fy0;
	    }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
#endif

/*
 * **************   M-DIM Parallelizaion **************************************
 * NOTE: Since we have to split the loop .... using omp for would be costly
 * We must have to manage it manually..... 
 * We need to urnoll the outer loop as much as possible:  8*VLEN = 64 
 *    May need to write intrinsic code to make sure vectoried.. though compiler
 *    is not bad to vectorize following code
 * ToDO: manage it manually........ 
 */
   #define DEBUG 0 
      #define DOUT stderr     // debug output 

   //#define MDIM_UR8_NOSYNC 1  // N=256: spup = 0.572 to NDIM Efficient vecrion  
                              // N=128 spup = 0.634    
                              // N=64: spup = 0.683  
                              // N=32: spup = 0.974  

   //#define MDIM_UR4 1  // spup = 0.278   
   
   //#define MDIM_VEC_UR8 1  // N=256: spup = 1.46  
                           // N=128: spup = 1.56
                           // N=64: spup = 1.85
   
   //#define MDIM_VEC_UR4 1  // N=256: spup = 1.35 
                           // N=128: spup = 1.42  
                           // N=64: spup = 1.46  
                           // N=32: spup = 2.03  --- with 256 Eff, spup = 1.26   
   
   //#define MDIM_VEC_UR8_NOSYNC 1  // N=256: spup = 1.93  .... FIXME: energy=nan 
                                  // N = 128: spup = 2.07 
                                  // N = 64: spup = 2.27  
   
   #define MDIM_VEC_UR4_NOSYNC 1  // N=256: spup = 1.8 ... FIXME: energy=nan 
                                  // N=128: spup = 1.9
                                  // N=64: spup = 2.25 -- with 256 eff, spup = 1.85  
                                  // N=32: spup = 2.91 -- with 256 eff, spup = 1.89 
   
   #ifdef MDIM_UR8 

   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7;
               VALUETYPE fy0, fy1, fy2, fy3, fy4, fy5, fy6, fy7;
               
               VALUETYPE x0, x1, x2, x3, x4, x5, x6, x7;
               VALUETYPE y0, y1, y2, y3, y4, y5, y6, y7;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];
	       x4 = blasX[i+4];
	       x5 = blasX[i+5];
	       x6 = blasX[i+6];
	       x7 = blasX[i+7];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
	       y4 = blasY[i+4];
	       y5 = blasY[i+5];
	       y6 = blasY[i+6];
	       y7 = blasY[i+7];
					
	       fx0 = fx1 = fx2 = fx3 = fx4 = fx5 = fx6 = fx7 = 0;
	       fy0 = fy1 = fy2 = fy3 = fy4 = fy5 = fy6 = fy7 = 0;		
	
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  VALUETYPE dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7;
                  VALUETYPE dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7;
                  VALUETYPE d0, d1, d2, d3, d4, d5, d6, d7;
                  
                  VALUETYPE tfx0=0.0, tfx1=0.0, tfx2=0.0, tfx3=0.0, tfx4=0.0, 
                            tfx5=0.0, tfx6=0.0, tfx7=0.0;
                  VALUETYPE tfy0=0.0, tfy1=0.0, tfy2=0.0, tfy3=0.0, tfy4=0.0, 
                            tfy5=0.0, tfy6=0.0, tfy7=0.0;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();

/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
#if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "**********\n");
                     fprintf(DOUT, "lower half case: chunksize = %ld\n", 
                           chunksize );
                     fflush(DOUT);
                  }
#endif
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
		     dx4 = xj - x4;
                     dx5 = xj - x5;
                     dx6 = xj - x6;
                     dx7 = xj - x7;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		     dy4 = yj - y4;
                     dy5 = yj - y5;
                     dy6 = yj - y6;
                     dy7 = yj - y7; 	       
		
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		     tfx4 += dx4 * d4;
                     tfx5 += dx5 * d5;
                     tfx6 += dx6 * d6;
                     tfx7 += dx7 * d7;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
		     tfy4 += dy4 * d4;
                     tfy5 += dy5 * d5;
                     tfy6 += dy6 * d6;
                     tfy7 += dy7 * d7;
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d: lower cleanup case = %d\n", i, 
                           i-nthreads*chunksize); 
         #endif
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
		        dx4 = xj - x4;
                        dx5 = xj - x5;
                        dx6 = xj - x6;
                        dx7 = xj - x7;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		        dy4 = yj - y4;
                        dy5 = yj - y5;
                        dy6 = yj - y6;
                        dy7 = yj - y7; 	       
		
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		        d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		        d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		        d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		        tfx4 += dx4 * d4;
                        tfx5 += dx5 * d5;
                        tfx6 += dx6 * d6;
                        tfx7 += dx7 * d7;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
		        tfy4 += dy4 * d4;
                        tfy5 += dy5 * d5;
                        tfy6 += dy6 * d6;
                        tfy7 += dy7 * d7;
                     }
                  //}

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                  //#pragma omp single nowait
                  //{
                     for (j = i; j < i+8; j++) 
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
                        if (j != i)
                        {
                           VALUETYPE dx0, dy0;
                           dx0 = xj - x0;
                           dy0 = yj - y0;
	                   d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		           tfx0 += dx0 * d0;
                           tfy0 += dy0 * d0;
                        }
                        if ( j != i+1 )
                        {
                           VALUETYPE dx1, dy1;
                           dx1 = xj - x1;
                           dy1 = yj - y1;
	                   d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		           tfx1 += dx1 * d1;
                           tfy1 += dy1 * d1;
                        }
                        if ( j != i+2 )
                        {
                           VALUETYPE dx2, dy2;
                           dx2 = xj - x2;
                           dy2 = yj - y2;
	                   d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		           tfx2 += dx2 * d2;
                           tfy2 += dy2 * d2;
                        }
                        if ( j != i+3 )
                        {
                           VALUETYPE dx3, dy3;
                           dx3 = xj - x3;
                           dy3 = yj - y3;
	                   d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		           tfx3 += dx3 * d3;
                           tfy3 += dy3 * d3;
                        }
                        if ( j != i+4 )
                        {
                           VALUETYPE dx4, dy4;
                           dx4 = xj - x4;
                           dy4 = yj - y4;
	                   d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		           tfx4 += dx4 * d4;
                           tfy4 += dy4 * d4;
                        }
                        if ( j != i+5 )
                        {
                           VALUETYPE dx5, dy5;
                           dx5 = xj - x5;
                           dy5 = yj - y5;
	                   d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		           tfx5 += dx5 * d5;
                           tfy5 += dy5 * d5;
                        }
                        if ( j != i+6 )
                        {
                           VALUETYPE dx6, dy6;
                           dx6 = xj - x6;
                           dy6 = yj - y6;
	                   d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		           tfx6 += dx6 * d6;
                           tfy6 += dy6 * d6;
                        }
                        if ( j != i+7 )
                        {
                           VALUETYPE dx7, dy7;
                           dx7 = xj - x7;
                           dy7 = yj - y7;
	                   d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);
		           tfx7 += dx7 * d7;
                           tfy7 += dy7 * d7;
                        }
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - 8)/nthreads; 
         #if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "upper half case: chunksize = %d\n", 
                           chunksize);
                  }
         #endif

                  for(j = (i+8)+id*chunksize; j < (i+8)+(id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
		     dx4 = xj - x4;
                     dx5 = xj - x5;
                     dx6 = xj - x6;
                     dx7 = xj - x7;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		     dy4 = yj - y4;
                     dy5 = yj - y5;
                     dy6 = yj - y6;
                     dy7 = yj - y7; 	       
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		     tfx4 += dx4 * d4;
                     tfx5 += dx5 * d5;
                     tfx6 += dx6 * d6;
                     tfx7 += dx7 * d7;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
		     tfy4 += dy4 * d4;
                     tfy5 += dy5 * d5;
                     tfy6 += dy6 * d6;
                     tfy7 += dy7 * d7;
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d (id=%d): lower cleanup case\n", i, id); 
         #endif
                     for (j=(i+8)+nthreads*chunksize; j < M; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
		        dx4 = xj - x4;
                        dx5 = xj - x5;
                        dx6 = xj - x6;
                        dx7 = xj - x7;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		        dy4 = yj - y4;
                        dy5 = yj - y5;
                        dy6 = yj - y6;
                        dy7 = yj - y7; 	       
                                   	
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		        d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		        d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		        d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		        tfx4 += dx4 * d4;
                        tfx5 += dx5 * d5;
                        tfx6 += dx6 * d6;
                        tfx7 += dx7 * d7;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
		        tfy4 += dy4 * d4;
                        tfy5 += dy5 * d5;
                        tfy6 += dy6 * d6;
                        tfy7 += dy7 * d7;
                     }
                  }
/*
 *                Reduction 
 */
                  #pragma omp critical
                  {
                     fx0 += tfx0; 
                     fx1 += tfx1; 
                     fx2 += tfx2; 
                     fx3 += tfx3; 
                     fx4 += tfx4; 
                     fx5 += tfx5; 
                     fx6 += tfx6; 
                     fx7 += tfx7; 
                     
                     fy0 += tfy0; 
                     fy1 += tfy1; 
                     fy2 += tfy2; 
                     fy3 += tfy3; 
                     fy4 += tfy4; 
                     fy5 += tfy5; 
                     fy6 += tfy6; 
                     fy7 += tfy7; 
                  }
               }
/*
 *             Out of parallel region 
 */
               pb_X[ind] = fx0;
	       pb_X[ind+1] = fx1;
	       pb_X[ind+2] = fx2;
               pb_X[ind+3] = fx3;
	       pb_X[ind+4] = fx4;
               pb_X[ind+5] = fx5;
               pb_X[ind+6] = fx6;
               pb_X[ind+7] = fx7;
					
	       pb_Y[ind] = fy0;
	       pb_Y[ind+1] = fy1;
	       pb_Y[ind+2] = fy2;
               pb_Y[ind+3] = fy3;
	       pb_Y[ind+4] = fy4;
               pb_Y[ind+5] = fy5;
               pb_Y[ind+6] = fy6;
               pb_Y[ind+7] = fy7;	
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
/*
 *    Without critical section 
 */

#elif defined (MDIM_UR8_NOSYNC) 
   

   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
     
#if 0
      cout << "Rows = " << graph.rows << endl;
      exit(1);
#endif

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 8)
            {
	       int ind = i-b*BATCHSIZE;
               int m, n;
               double sum[NUMOFTHREADS][2][8]; // 8+8 sum for each threads, one cache line 

               VALUETYPE fx0, fx1, fx2, fx3, fx4, fx5, fx6, fx7;
               VALUETYPE fy0, fy1, fy2, fy3, fy4, fy5, fy6, fy7;
               
               VALUETYPE x0, x1, x2, x3, x4, x5, x6, x7;
               VALUETYPE y0, y1, y2, y3, y4, y5, y6, y7;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];
	       x4 = blasX[i+4];
	       x5 = blasX[i+5];
	       x6 = blasX[i+6];
	       x7 = blasX[i+7];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
	       y4 = blasY[i+4];
	       y5 = blasY[i+5];
	       y6 = blasY[i+6];
	       y7 = blasY[i+7];
					
	       fx0 = fx1 = fx2 = fx3 = fx4 = fx5 = fx6 = fx7 = 0;
	       fy0 = fy1 = fy2 = fy3 = fy4 = fy5 = fy6 = fy7 = 0;		
               
               // init sum mem
               for (m=0; m < NUMOFTHREADS; m++)
                  for (n=0; n < 8; n++)
                     sum[m][0][n] = sum[m][1][n] = 0.0;

               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  VALUETYPE dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7;
                  VALUETYPE dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7;
                  VALUETYPE d0, d1, d2, d3, d4, d5, d6, d7;
                  
                  VALUETYPE tfx0=0.0, tfx1=0.0, tfx2=0.0, tfx3=0.0, tfx4=0.0, 
                            tfx5=0.0, tfx6=0.0, tfx7=0.0;
                  VALUETYPE tfy0=0.0, tfy1=0.0, tfy2=0.0, tfy3=0.0, tfy4=0.0, 
                            tfy5=0.0, tfy6=0.0, tfy7=0.0;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();

/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
#if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "**********\n");
                     fprintf(DOUT, "lower half case: chunksize = %ld\n", 
                           chunksize );
                     fflush(DOUT);
                  }
#endif
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
		     dx4 = xj - x4;
                     dx5 = xj - x5;
                     dx6 = xj - x6;
                     dx7 = xj - x7;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		     dy4 = yj - y4;
                     dy5 = yj - y5;
                     dy6 = yj - y6;
                     dy7 = yj - y7; 	       
		
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		     tfx4 += dx4 * d4;
                     tfx5 += dx5 * d5;
                     tfx6 += dx6 * d6;
                     tfx7 += dx7 * d7;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
		     tfy4 += dy4 * d4;
                     tfy5 += dy5 * d5;
                     tfy6 += dy6 * d6;
                     tfy7 += dy7 * d7;
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d: lower cleanup case = %d\n", i, 
                           i-nthreads*chunksize); 
         #endif
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
		        dx4 = xj - x4;
                        dx5 = xj - x5;
                        dx6 = xj - x6;
                        dx7 = xj - x7;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		        dy4 = yj - y4;
                        dy5 = yj - y5;
                        dy6 = yj - y6;
                        dy7 = yj - y7; 	       
		
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		        d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		        d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		        d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		        tfx4 += dx4 * d4;
                        tfx5 += dx5 * d5;
                        tfx6 += dx6 * d6;
                        tfx7 += dx7 * d7;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
		        tfy4 += dy4 * d4;
                        tfy5 += dy5 * d5;
                        tfy6 += dy6 * d6;
                        tfy7 += dy7 * d7;
                     }
                  //}

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                  //#pragma omp single nowait
                  //{
                     for (j = i; j < i+8; j++) 
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
                        if (j != i)
                        {
                           VALUETYPE dx0, dy0;
                           dx0 = xj - x0;
                           dy0 = yj - y0;
	                   d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		           tfx0 += dx0 * d0;
                           tfy0 += dy0 * d0;
                        }
                        if ( j != i+1 )
                        {
                           VALUETYPE dx1, dy1;
                           dx1 = xj - x1;
                           dy1 = yj - y1;
	                   d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		           tfx1 += dx1 * d1;
                           tfy1 += dy1 * d1;
                        }
                        if ( j != i+2 )
                        {
                           VALUETYPE dx2, dy2;
                           dx2 = xj - x2;
                           dy2 = yj - y2;
	                   d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		           tfx2 += dx2 * d2;
                           tfy2 += dy2 * d2;
                        }
                        if ( j != i+3 )
                        {
                           VALUETYPE dx3, dy3;
                           dx3 = xj - x3;
                           dy3 = yj - y3;
	                   d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		           tfx3 += dx3 * d3;
                           tfy3 += dy3 * d3;
                        }
                        if ( j != i+4 )
                        {
                           VALUETYPE dx4, dy4;
                           dx4 = xj - x4;
                           dy4 = yj - y4;
	                   d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		           tfx4 += dx4 * d4;
                           tfy4 += dy4 * d4;
                        }
                        if ( j != i+5 )
                        {
                           VALUETYPE dx5, dy5;
                           dx5 = xj - x5;
                           dy5 = yj - y5;
	                   d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		           tfx5 += dx5 * d5;
                           tfy5 += dy5 * d5;
                        }
                        if ( j != i+6 )
                        {
                           VALUETYPE dx6, dy6;
                           dx6 = xj - x6;
                           dy6 = yj - y6;
	                   d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		           tfx6 += dx6 * d6;
                           tfy6 += dy6 * d6;
                        }
                        if ( j != i+7 )
                        {
                           VALUETYPE dx7, dy7;
                           dx7 = xj - x7;
                           dy7 = yj - y7;
	                   d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);
		           tfx7 += dx7 * d7;
                           tfy7 += dy7 * d7;
                        }
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - 8)/nthreads; 
         #if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "upper half case: chunksize = %d\n", 
                           chunksize);
                  }
         #endif

                  for(j = (i+8)+id*chunksize; j < (i+8)+(id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
		     dx4 = xj - x4;
                     dx5 = xj - x5;
                     dx6 = xj - x6;
                     dx7 = xj - x7;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		     dy4 = yj - y4;
                     dy5 = yj - y5;
                     dy6 = yj - y6;
                     dy7 = yj - y7; 	       
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		     d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		     d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		     d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		     d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		     tfx4 += dx4 * d4;
                     tfx5 += dx5 * d5;
                     tfx6 += dx6 * d6;
                     tfx7 += dx7 * d7;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
		     tfy4 += dy4 * d4;
                     tfy5 += dy5 * d5;
                     tfy6 += dy6 * d6;
                     tfy7 += dy7 * d7;
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d (id=%d): lower cleanup case\n", i, id); 
         #endif
                     for (j=(i+8)+nthreads*chunksize; j < M; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
		        dx4 = xj - x4;
                        dx5 = xj - x5;
                        dx6 = xj - x6;
                        dx7 = xj - x7;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		        dy4 = yj - y4;
                        dy5 = yj - y5;
                        dy6 = yj - y6;
                        dy7 = yj - y7; 	       
                                   	
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		        d4 = 1.0 / (dx4 * dx4 + dy4 * dy4);
		        d5 = 1.0 / (dx5 * dx5 + dy5 * dy5);
		        d6 = 1.0 / (dx6 * dx6 + dy6 * dy6);
		        d7 = 1.0 / (dx7 * dx7 + dy7 * dy7);       
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		        tfx4 += dx4 * d4;
                        tfx5 += dx5 * d5;
                        tfx6 += dx6 * d6;
                        tfx7 += dx7 * d7;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
		        tfy4 += dy4 * d4;
                        tfy5 += dy5 * d5;
                        tfy6 += dy6 * d6;
                        tfy7 += dy7 * d7;
                     }
                  }
/*
 *                Reduction using memory  
 */
                  {
                     sum[id][0][0] += tfx0; 
                     sum[id][0][1] += tfx1; 
                     sum[id][0][2] += tfx2; 
                     sum[id][0][3] += tfx3; 
                     sum[id][0][4] += tfx4; 
                     sum[id][0][5] += tfx5; 
                     sum[id][0][6] += tfx6; 
                     sum[id][0][7] += tfx7; 
                     
                     sum[id][1][0] += tfy0; 
                     sum[id][1][1] += tfy1; 
                     sum[id][1][2] += tfy2; 
                     sum[id][1][3] += tfy3; 
                     sum[id][1][4] += tfy4; 
                     sum[id][1][5] += tfy5; 
                     sum[id][1][6] += tfy6; 
                     sum[id][1][7] += tfy7; 
                  }
               }
/*
 *             Out of parallel region 
 */
/*
 *             sumup all the sums 
 */
               for (m=0; m < NUMOFTHREADS; m++)
               {
                  fx0 += sum[m][0][0]; 
                  fx1 += sum[m][0][1]; 
                  fx2 += sum[m][0][2]; 
                  fx3 += sum[m][0][3]; 
                  fx4 += sum[m][0][4]; 
                  fx5 += sum[m][0][5]; 
                  fx6 += sum[m][0][6]; 
                  fx7 += sum[m][0][7]; 
                  
                  fy0 += sum[m][1][0]; 
                  fy1 += sum[m][1][1]; 
                  fy2 += sum[m][1][2]; 
                  fy3 += sum[m][1][3]; 
                  fy4 += sum[m][1][4]; 
                  fy5 += sum[m][1][5]; 
                  fy6 += sum[m][1][6]; 
                  fy7 += sum[m][1][7]; 
               }


               pb_X[ind] = fx0;
	       pb_X[ind+1] = fx1;
	       pb_X[ind+2] = fx2;
               pb_X[ind+3] = fx3;
	       pb_X[ind+4] = fx4;
               pb_X[ind+5] = fx5;
               pb_X[ind+6] = fx6;
               pb_X[ind+7] = fx7;
					
	       pb_Y[ind] = fy0;
	       pb_Y[ind+1] = fy1;
	       pb_Y[ind+2] = fy2;
               pb_Y[ind+3] = fy3;
	       pb_Y[ind+4] = fy4;
               pb_Y[ind+5] = fy5;
               pb_Y[ind+6] = fy6;
               pb_Y[ind+7] = fy7;	
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }


#elif defined (MDIM_UR4) 

   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
   

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 4)
            {
	       int ind = i-b*BATCHSIZE;

               VALUETYPE fx0, fx1, fx2, fx3;
               VALUETYPE fy0, fy1, fy2, fy3;
               
               VALUETYPE x0, x1, x2, x3, x4;
               VALUETYPE y0, y1, y2, y3, y4;
					
	       x0 = blasX[i];
	       x1 = blasX[i+1];
	       x2 = blasX[i+2];
	       x3 = blasX[i+3];

	       y0 = blasY[i];
	       y1 = blasY[i+1];	
	       y2 = blasY[i+2];
	       y3 = blasY[i+3];
					
	       fx0 = fx1 = fx2 = fx3 = 0.0;
	       fy0 = fy1 = fy2 = fy3 = 0.0;		
	
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  VALUETYPE dx0, dx1, dx2, dx3;
                  VALUETYPE dy0, dy1, dy2, dy3;
                  VALUETYPE d0, d1, d2, d3;
                  
                  VALUETYPE tfx0=0.0, tfx1=0.0, tfx2=0.0, tfx3=0.0;
                  VALUETYPE tfy0=0.0, tfy1=0.0, tfy2=0.0, tfy3=0.0;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();

/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
#if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "**********\n");
                     fprintf(DOUT, "lower half case: chunksize = %ld\n", 
                           chunksize );
                     fflush(DOUT);
                  }
#endif
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
		
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d: lower cleanup case = %d\n", i, 
                           i-nthreads*chunksize); 
         #endif
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
		
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
                     }
/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                     for (j = i; j < i+4; j++) 
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
                        if (j != i)
                        {
                           VALUETYPE dx0, dy0;
                           dx0 = xj - x0;
                           dy0 = yj - y0;
	                   d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		           tfx0 += dx0 * d0;
                           tfy0 += dy0 * d0;
                        }
                        if ( j != i+1 )
                        {
                           VALUETYPE dx1, dy1;
                           dx1 = xj - x1;
                           dy1 = yj - y1;
	                   d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		           tfx1 += dx1 * d1;
                           tfy1 += dy1 * d1;
                        }
                        if ( j != i+2 )
                        {
                           VALUETYPE dx2, dy2;
                           dx2 = xj - x2;
                           dy2 = yj - y2;
	                   d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		           tfx2 += dx2 * d2;
                           tfy2 += dy2 * d2;
                        }
                        if ( j != i+3 )
                        {
                           VALUETYPE dx3, dy3;
                           dx3 = xj - x3;
                           dy3 = yj - y3;
	                   d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		           tfx3 += dx3 * d3;
                           tfy3 += dy3 * d3;
                        }
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - 4)/nthreads; 
         #if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "upper half case: chunksize = %d\n", 
                           chunksize);
                  }
         #endif

                  for(j = (i+4)+id*chunksize; j < (i+4)+(id+1)*chunksize; j++)
                  {
                     VALUETYPE xj = blasX[j];
                     VALUETYPE yj = blasY[j];
		  
                     dx0 = xj - x0;
		     dx1 = xj - x1;
		     dx2 = xj - x2;
                     dx3 = xj - x3;
	          
                     dy0 = yj - y0;
		     dy1 = yj - y1;
		     dy2 = yj - y2;
                     dy3 = yj - y3;
                                   	
	             d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		     d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		     d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		     tfx0 += dx0 * d0;
		     tfx1 += dx1 * d1;
		     tfx2 += dx2 * d2;
                     tfx3 += dx3 * d3;
		
                     tfy0 += dy0 * d0;
		     tfy1 += dy1 * d1;
		     tfy2 += dy2 * d2;
                     tfy3 += dy3 * d3;
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d (id=%d): lower cleanup case\n", i, id); 
         #endif
                     for (j=(i+4)+nthreads*chunksize; j < M; j++)
                     {
                        VALUETYPE xj = blasX[j];
                        VALUETYPE yj = blasY[j];
		  
                        dx0 = xj - x0;
		        dx1 = xj - x1;
		        dx2 = xj - x2;
                        dx3 = xj - x3;
	          
                        dy0 = yj - y0;
		        dy1 = yj - y1;
		        dy2 = yj - y2;
                        dy3 = yj - y3;
                                   	
	                d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        d1 = 1.0 / (dx1 * dx1 + dy1 * dy1);
		        d2 = 1.0 / (dx2 * dx2 + dy2 * dy2);
		        d3 = 1.0 / (dx3 * dx3 + dy3 * dy3);
		
		        tfx0 += dx0 * d0;
		        tfx1 += dx1 * d1;
		        tfx2 += dx2 * d2;
                        tfx3 += dx3 * d3;
		
                        tfy0 += dy0 * d0;
		        tfy1 += dy1 * d1;
		        tfy2 += dy2 * d2;
                        tfy3 += dy3 * d3;
                     }
                  }
/*
 *                Reduction 
 */
                  #pragma omp critical
                  {
                     fx0 += tfx0; 
                     fx1 += tfx1; 
                     fx2 += tfx2; 
                     fx3 += tfx3; 
                     
                     fy0 += tfy0; 
                     fy1 += tfy1; 
                     fy2 += tfy2; 
                     fy3 += tfy3; 
                  }
               }
/*
 *             Out of parallel region 
 */
               pb_X[ind] = fx0;
	       pb_X[ind+1] = fx1;
	       pb_X[ind+2] = fx2;
               pb_X[ind+3] = fx3;
					
	       pb_Y[ind] = fy0;
	       pb_Y[ind+1] = fy1;
	       pb_Y[ind+2] = fy2;
               pb_Y[ind+3] = fy3;
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
   }

#elif defined (MDIM_VEC_UR8)
   
   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();

/*
 *    NOTE: we have different version of reciprocals... I'm defining it here
 *                   NOTE: rcp14_pd inst... precision 2^-14
 *                   rcp28_pd has relative error 2^-28 ... rounding??? 
 *                   NOTE: reciprocal is highly error prone for over/underflow
 *                   FIXME: to compare apple to apple, make div version 
 */
      #define _MM512_RCP_PD _mm512_rcp14_pd  // rcp28 -- AVX512ER 
      #define _MM512_MASKZ_RCP_PD _mm512_maskz_rcp14_pd  // rcp28

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 64)
            {
	       int ind = i-b*BATCHSIZE;
               register __m512d vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7;
               register __m512d vy0, vy1, vy2, vy3, vy4, vy5, vy6, vy7;

               __m512d vfx0, vfx1, vfx2, vfx3, vfx4, vfx5, vfx6, vfx7;
               __m512d vfy0, vfy1, vfy2, vfy3, vfy4, vfy5, vfy6, vfy7;
					
	       vx0 = _mm512_loadu_pd(blasX + i);
	       vx1 = _mm512_loadu_pd(blasX + i+8);
	       vx2 = _mm512_loadu_pd(blasX + i+2*8);
	       vx3 = _mm512_loadu_pd(blasX + i+3*8);
	       vx4 = _mm512_loadu_pd(blasX + i+4*8);
	       vx5 = _mm512_loadu_pd(blasX + i+5*8);
	       vx6 = _mm512_loadu_pd(blasX + i+6*8);
	       vx7 = _mm512_loadu_pd(blasX + i+7*8);

	       vy0 = _mm512_loadu_pd(blasY + i);
	       vy1 = _mm512_loadu_pd(blasY + i+1*8);
	       vy2 = _mm512_loadu_pd(blasY + i+2*8);
	       vy3 = _mm512_loadu_pd(blasY + i+3*8);
	       vy4 = _mm512_loadu_pd(blasY + i+4*8);
	       vy5 = _mm512_loadu_pd(blasY + i+5*8);
	       vy6 = _mm512_loadu_pd(blasY + i+6*8);
	       vy7 = _mm512_loadu_pd(blasY + i+7*8);
					
	       vfx0 = _mm512_set1_pd(0.0);
	       vfx1 = _mm512_set1_pd(0.0);
	       vfx2 = _mm512_set1_pd(0.0);
	       vfx3 = _mm512_set1_pd(0.0);
	       vfx4 = _mm512_set1_pd(0.0);
	       vfx5 = _mm512_set1_pd(0.0);
	       vfx6 = _mm512_set1_pd(0.0);
	       vfx7 = _mm512_set1_pd(0.0);
	
	       vfy0 = _mm512_set1_pd(0.0);
	       vfy1 = _mm512_set1_pd(0.0);
	       vfy2 = _mm512_set1_pd(0.0);
	       vfy3 = _mm512_set1_pd(0.0);
	       vfy4 = _mm512_set1_pd(0.0);
	       vfy5 = _mm512_set1_pd(0.0);
	       vfy6 = _mm512_set1_pd(0.0);
	       vfy7 = _mm512_set1_pd(0.0);
               
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  __m512d vdx0, vdx1, vdx2, vdx3, vdx4, vdx5, vdx6, vdx7;
                  __m512d vdy0, vdy1, vdy2, vdy3, vdy4, vdy5, vdy6, vdy7;
                  __m512d vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7;
/*
 *                NOTE: don't need those temp regs if no_sync used 
 */
                  __m512d vtfx0, vtfx1, vtfx2, vtfx3, vtfx4, vtfx5, vtfx6, vtfx7;
                  __m512d vtfy0, vtfy1, vtfy2, vtfy3, vtfy4, vtfy5, vtfy6, vtfy7;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();
                  
                  // init temp sum 
		  vtfx0 = _mm512_set1_pd(0.0); vtfy0 = _mm512_set1_pd(0.0);
		  vtfx1 = _mm512_set1_pd(0.0); vtfy1 = _mm512_set1_pd(0.0);
		  vtfx2 = _mm512_set1_pd(0.0); vtfy2 = _mm512_set1_pd(0.0);
		  vtfx3 = _mm512_set1_pd(0.0); vtfy3 = _mm512_set1_pd(0.0);
		  vtfx4 = _mm512_set1_pd(0.0); vtfy4 = _mm512_set1_pd(0.0);
		  vtfx5 = _mm512_set1_pd(0.0); vtfy5 = _mm512_set1_pd(0.0);
		  vtfx6 = _mm512_set1_pd(0.0); vtfy6 = _mm512_set1_pd(0.0);
		  vtfx7 = _mm512_set1_pd(0.0); vtfy7 = _mm512_set1_pd(0.0);
/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     double xj = blasX[j];
                     register __m512d vxj = _mm512_set1_pd(xj);
                     double yj = blasY[j];
                     register __m512d vyj = _mm512_set1_pd(yj);
		  
                     //dx0 = xj - x0;
		     vdx0 = _mm512_sub_pd(vxj, vx0);
		     vdx1 = _mm512_sub_pd(vxj, vx1);
		     vdx2 = _mm512_sub_pd(vxj, vx2);
		     vdx3 = _mm512_sub_pd(vxj, vx3);
		     vdx4 = _mm512_sub_pd(vxj, vx4);
		     vdx5 = _mm512_sub_pd(vxj, vx5);
		     vdx6 = _mm512_sub_pd(vxj, vx6);
		     vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                     // dy0 = yj - y0;
		     vdy0 = _mm512_sub_pd(vyj, vy0);
		     vdy1 = _mm512_sub_pd(vyj, vy1);
		     vdy2 = _mm512_sub_pd(vyj, vy2);
		     vdy3 = _mm512_sub_pd(vyj, vy3);
		     vdy4 = _mm512_sub_pd(vyj, vy4);
		     vdy5 = _mm512_sub_pd(vyj, vy5);
		     vdy6 = _mm512_sub_pd(vyj, vy6);
		     vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     vd0 = _mm512_mul_pd(vdx0, vdx0);
                     vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                     vd0 = _MM512_RCP_PD(vd0); 
                     
		     vd1 = _mm512_mul_pd(vdx1, vdx1);
                     vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                     vd1 = _MM512_RCP_PD(vd1); 
                     
		     vd2 = _mm512_mul_pd(vdx2, vdx2);
                     vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                     vd2 = _MM512_RCP_PD(vd2); 
                     
		     vd3 = _mm512_mul_pd(vdx3, vdx3);
                     vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                     vd3 = _MM512_RCP_PD(vd3); 
                  
		     vd4 = _mm512_mul_pd(vdx4, vdx4);
                     vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                     vd4 = _MM512_RCP_PD(vd4); 
                     
                     
		     vd5 = _mm512_mul_pd(vdx5, vdx5);
                     vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                     vd5 = _MM512_RCP_PD(vd5); 
                     
                     
		     vd6 = _mm512_mul_pd(vdx6, vdx6);
                     vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                     vd6 = _MM512_RCP_PD(vd6); 
                     
		     vd7 = _mm512_mul_pd(vdx7, vdx7);
                     vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                     vd7 = _MM512_RCP_PD(vd7); 
		
		     
                     //tfx0 += dx0 * d0;
                     vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                     vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                     vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                     vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                     vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                     vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                     vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                     vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                     //tfy0 += dy0 * d0;
                     vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                     vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                     vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                     vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                     vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                     vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                     vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                     vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d: lower cleanup case = %d\n", i, 
                           i-nthreads*chunksize); 
         #endif
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
		  
                        //dx0 = xj - x0;
		        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
		        vdx4 = _mm512_sub_pd(vxj, vx4);
		        vdx5 = _mm512_sub_pd(vxj, vx5);
		        vdx6 = _mm512_sub_pd(vxj, vx6);
		        vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
		        vdy4 = _mm512_sub_pd(vyj, vy4);
		        vdy5 = _mm512_sub_pd(vyj, vy5);
		        vdy6 = _mm512_sub_pd(vyj, vy6);
		        vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        vd0 = _MM512_RCP_PD(vd0); 
                     
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd1 = _MM512_RCP_PD(vd1); 
                     
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        vd2 = _MM512_RCP_PD(vd2); 
                     
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        vd3 = _MM512_RCP_PD(vd3); 
                  
		        vd4 = _mm512_mul_pd(vdx4, vdx4);
                        vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                        vd4 = _MM512_RCP_PD(vd4); 
                     
		        vd5 = _mm512_mul_pd(vdx5, vdx5);
                        vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                        vd5 = _MM512_RCP_PD(vd5); 
                     
		        vd6 = _mm512_mul_pd(vdx6, vdx6);
                        vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                        vd6 = _MM512_RCP_PD(vd6); 
                     
		        vd7 = _mm512_mul_pd(vdx7, vdx7);
                        vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                        vd7 = _MM512_RCP_PD(vd7); 
		     
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                        vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                        vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                        vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                        vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                        vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                        vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                        vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                        vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
                     }

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                     for (j = i; j < i+64; j++) 
                     {
                        unsigned int n = j-i;
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
                        register __m512d vzero = _mm512_set1_pd(0);
                        
                        // by default all 1 (for 8 elements)
                        unsigned int ik0, ik1, ik2, ik3, ik4, ik5, ik6, ik7; 
                        __mmask8 k0, k1, k2, k3, k4, k5, k6, k7; 
            #if 1
/*
 *                      Possible to further optimize.. will think later 
 */
                        ik0 = ( n >= 0 && n < 8 ) ? (255 & (~(1 << (n%8))))  : 255; 
                        ik1 = ( n >=8 && n < 8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik2 = ( n >=2*8 && n < 2*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik3 = ( n >=3*8 && n < 3*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik4 = ( n >=4*8 && n < 4*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik5 = ( n >=5*8 && n < 5*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik6 = ( n >=6*8 && n < 6*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik7 = ( n >=7*8 && n < 7*8+8 )? (255 & ~(1 << n%8))  : 255; 
		        
                        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
		        vdx4 = _mm512_sub_pd(vxj, vx4);
		        vdx5 = _mm512_sub_pd(vxj, vx5);
		        vdx6 = _mm512_sub_pd(vxj, vx6);
		        vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
		        vdy4 = _mm512_sub_pd(vyj, vy4);
		        vdy5 = _mm512_sub_pd(vyj, vy5);
		        vdy6 = _mm512_sub_pd(vyj, vy6);
		        vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        k0 = _cvtu32_mask8(ik0);
                        //vd0 = _MM512_MASKZ_RCP_PD(vd0, k0, vd0); 
                        vd0 = _MM512_MASKZ_RCP_PD(k0, vd0); 
                     
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd1 = _MM512_RCP_PD(vd1); 
                        k1 = _cvtu32_mask8(ik1);
                        //vd1 = _MM512_MASKZ_RCP_PD(vd1, k1, vd1); 
                        vd1 = _MM512_MASKZ_RCP_PD(k1, vd1); 
                     
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        k2 = _cvtu32_mask8(ik2);
                        //vd2 = _MM512_MASKZ_RCP_PD(vd2, k2, vd2); 
                        vd2 = _MM512_MASKZ_RCP_PD(k2, vd2); 
                     
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        k3 = _cvtu32_mask8(ik3);
                        //vd3 = _MM512_MASKZ_RCP_PD(vd3, k3, vd3); 
                        vd3 = _MM512_MASKZ_RCP_PD(k3, vd3); 
                  
		        vd4 = _mm512_mul_pd(vdx4, vdx4);
                        vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                        k4 = _cvtu32_mask8(ik4);
                        //vd4 = _MM512_MASKZ_RCP_PD(vd4, k4, vd4); 
                        vd4 = _MM512_MASKZ_RCP_PD(k4, vd4); 
                     
		        vd5 = _mm512_mul_pd(vdx5, vdx5);
                        vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                        k5 = _cvtu32_mask8(ik5);
                        //vd5 = _MM512_MASKZ_RCP_PD(vd5, k5, vd5); 
                        vd5 = _MM512_MASKZ_RCP_PD(k5, vd5); 
                     
		        vd6 = _mm512_mul_pd(vdx6, vdx6);
                        vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                        k6 = _cvtu32_mask8(ik6);
                        //vd6 = _MM512_MASKZ_RCP_PD(vd6, k6, vd6); 
                        vd6 = _MM512_MASKZ_RCP_PD(k6, vd6); 
                     
		        vd7 = _mm512_mul_pd(vdx7, vdx7);
                        vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                        k7 = _cvtu32_mask8(ik7);
                        //vd7 = _MM512_MASKZ_RCP_PD(vd7, k7, vd7); 
                        vd7 = _MM512_MASKZ_RCP_PD(k7, vd7); 
		     
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                        vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                        vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                        vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                        vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                        vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                        vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                        vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                        vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
            #endif
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - 64)/nthreads; 
         #if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "upper half case: chunksize = %d\n", 
                           chunksize);
                  }
         #endif

                  for(j = (i+64)+id*chunksize; j < (i+64)+(id+1)*chunksize; j++)
                  {
                     double xj = blasX[j];
                     register __m512d vxj = _mm512_set1_pd(xj);
                     double yj = blasY[j];
                     register __m512d vyj = _mm512_set1_pd(yj);
		  
                     //dx0 = xj - x0;
		     vdx0 = _mm512_sub_pd(vxj, vx0);
		     vdx1 = _mm512_sub_pd(vxj, vx1);
		     vdx2 = _mm512_sub_pd(vxj, vx2);
		     vdx3 = _mm512_sub_pd(vxj, vx3);
		     vdx4 = _mm512_sub_pd(vxj, vx4);
		     vdx5 = _mm512_sub_pd(vxj, vx5);
		     vdx6 = _mm512_sub_pd(vxj, vx6);
		     vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                     // dy0 = yj - y0;
		     vdy0 = _mm512_sub_pd(vyj, vy0);
		     vdy1 = _mm512_sub_pd(vyj, vy1);
		     vdy2 = _mm512_sub_pd(vyj, vy2);
		     vdy3 = _mm512_sub_pd(vyj, vy3);
		     vdy4 = _mm512_sub_pd(vyj, vy4);
		     vdy5 = _mm512_sub_pd(vyj, vy5);
		     vdy6 = _mm512_sub_pd(vyj, vy6);
		     vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     vd0 = _mm512_mul_pd(vdx0, vdx0);
                     vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                     vd0 = _MM512_RCP_PD(vd0); 
                     
		     vd1 = _mm512_mul_pd(vdx1, vdx1);
                     vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                     vd1 = _MM512_RCP_PD(vd1); 
                     
		     vd2 = _mm512_mul_pd(vdx2, vdx2);
                     vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                     vd2 = _MM512_RCP_PD(vd2); 
                     
		     vd3 = _mm512_mul_pd(vdx3, vdx3);
                     vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                     vd3 = _MM512_RCP_PD(vd3); 
                  
		     vd4 = _mm512_mul_pd(vdx4, vdx4);
                     vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                     vd4 = _MM512_RCP_PD(vd4); 
                     
                     
		     vd5 = _mm512_mul_pd(vdx5, vdx5);
                     vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                     vd5 = _MM512_RCP_PD(vd5); 
                     
                     
		     vd6 = _mm512_mul_pd(vdx6, vdx6);
                     vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                     vd6 = _MM512_RCP_PD(vd6); 
                     
		     vd7 = _mm512_mul_pd(vdx7, vdx7);
                     vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                     vd7 = _MM512_RCP_PD(vd7); 
		
		     
                     //tfx0 += dx0 * d0;
                     vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                     vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                     vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                     vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                     vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                     vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                     vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                     vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                     //tfy0 += dy0 * d0;
                     vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                     vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                     vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                     vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                     vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                     vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                     vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                     vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d (id=%d): lower cleanup case\n", i, id); 
         #endif
                     for (j=(i+64)+nthreads*chunksize; j < M; j++)
                     {
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
		  
                        //dx0 = xj - x0;
		        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
		        vdx4 = _mm512_sub_pd(vxj, vx4);
		        vdx5 = _mm512_sub_pd(vxj, vx5);
		        vdx6 = _mm512_sub_pd(vxj, vx6);
		        vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
		        vdy4 = _mm512_sub_pd(vyj, vy4);
		        vdy5 = _mm512_sub_pd(vyj, vy5);
		        vdy6 = _mm512_sub_pd(vyj, vy6);
		        vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        vd0 = _MM512_RCP_PD(vd0); 
                     
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd1 = _MM512_RCP_PD(vd1); 
                     
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        vd2 = _MM512_RCP_PD(vd2); 
                     
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        vd3 = _MM512_RCP_PD(vd3); 
                  
		        vd4 = _mm512_mul_pd(vdx4, vdx4);
                        vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                        vd4 = _MM512_RCP_PD(vd4); 
                     
                     
		        vd5 = _mm512_mul_pd(vdx5, vdx5);
                        vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                        vd5 = _MM512_RCP_PD(vd5); 
                     
                     
		        vd6 = _mm512_mul_pd(vdx6, vdx6);
                        vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                        vd6 = _MM512_RCP_PD(vd6); 
                     
		        vd7 = _mm512_mul_pd(vdx7, vdx7);
                        vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                        vd7 = _MM512_RCP_PD(vd7); 
		
		     
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                        vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                        vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                        vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                        vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                        vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                        vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                        vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                        vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
                     }
                  }
/*
 *                Reduction 
 */
                  #pragma omp critical
                  {
                     //fx0 += tfx0; 
		     vfx0 = _mm512_add_pd(vfx0, vtfx0);
		     vfx1 = _mm512_add_pd(vfx1, vtfx1);
		     vfx2 = _mm512_add_pd(vfx2, vtfx2);
		     vfx3 = _mm512_add_pd(vfx3, vtfx3);
		     vfx4 = _mm512_add_pd(vfx4, vtfx4);
		     vfx5 = _mm512_add_pd(vfx5, vtfx5);
		     vfx6 = _mm512_add_pd(vfx6, vtfx6);
		     vfx7 = _mm512_add_pd(vfx7, vtfx7);
                     
                     //fx1 += tfx1; 
		     vfy0 = _mm512_add_pd(vfy0, vtfy0);
		     vfy1 = _mm512_add_pd(vfy1, vtfy1);
		     vfy2 = _mm512_add_pd(vfy2, vtfy2);
		     vfy3 = _mm512_add_pd(vfy3, vtfy3);
		     vfy4 = _mm512_add_pd(vfy4, vtfy4);
		     vfy5 = _mm512_add_pd(vfy5, vtfy5);
		     vfy6 = _mm512_add_pd(vfy6, vtfy6);
		     vfy7 = _mm512_add_pd(vfy7, vtfy7);
                  }
               }
/*
 *             Out of parallel region 
 */
               _mm512_storeu_pd(pb_X+ind, vfx0); 
               _mm512_storeu_pd(pb_X+ind+8, vfx1); 
               _mm512_storeu_pd(pb_X+ind+2*8, vfx2); 
               _mm512_storeu_pd(pb_X+ind+3*8, vfx3); 
               _mm512_storeu_pd(pb_X+ind+4*8, vfx4); 
               _mm512_storeu_pd(pb_X+ind+5*8, vfx5); 
               _mm512_storeu_pd(pb_X+ind+6*8, vfx6); 
               _mm512_storeu_pd(pb_X+ind+7*8, vfx7); 
               
               _mm512_storeu_pd(pb_Y+ind, vfy0); 
               _mm512_storeu_pd(pb_Y+ind+8, vfy1); 
               _mm512_storeu_pd(pb_Y+ind+2*8, vfy2); 
               _mm512_storeu_pd(pb_Y+ind+3*8, vfy3); 
               _mm512_storeu_pd(pb_Y+ind+4*8, vfy4); 
               _mm512_storeu_pd(pb_Y+ind+5*8, vfy5); 
               _mm512_storeu_pd(pb_Y+ind+6*8, vfy6); 
               _mm512_storeu_pd(pb_Y+ind+7*8, vfy7); 
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

#elif defined(MDIM_VEC_UR4)
   
   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();

/*
 *    NOTE: we have different version of reciprocals... I'm defining it here
 *                   NOTE: rcp14_pd inst... precision 2^-14
 *                   rcp28_pd has relative error 2^-28 ... rounding??? 
 *                   NOTE: reciprocal is highly error prone for over/underflow
 *                   FIXME: to compare apple to apple, make div version 
 */
      #define _MM512_RCP_PD _mm512_rcp14_pd  // rcp28 -- AVX512ER 
      #define _MM512_MASKZ_RCP_PD _mm512_maskz_rcp14_pd  // rcp28

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 32)
            {
	       int ind = i-b*BATCHSIZE;
               register __m512d vx0, vx1, vx2, vx3;
               register __m512d vy0, vy1, vy2, vy3;

               __m512d vfx0, vfx1, vfx2, vfx3;
               __m512d vfy0, vfy1, vfy2, vfy3;
					
	       vx0 = _mm512_loadu_pd(blasX + i);
	       vx1 = _mm512_loadu_pd(blasX + i+8);
	       vx2 = _mm512_loadu_pd(blasX + i+2*8);
	       vx3 = _mm512_loadu_pd(blasX + i+3*8);

	       vy0 = _mm512_loadu_pd(blasY + i);
	       vy1 = _mm512_loadu_pd(blasY + i+1*8);
	       vy2 = _mm512_loadu_pd(blasY + i+2*8);
	       vy3 = _mm512_loadu_pd(blasY + i+3*8);
					
	       vfx0 = _mm512_set1_pd(0.0);
	       vfx1 = _mm512_set1_pd(0.0);
	       vfx2 = _mm512_set1_pd(0.0);
	       vfx3 = _mm512_set1_pd(0.0);
	
	       vfy0 = _mm512_set1_pd(0.0);
	       vfy1 = _mm512_set1_pd(0.0);
	       vfy2 = _mm512_set1_pd(0.0);
	       vfy3 = _mm512_set1_pd(0.0);
               
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j, remainder; 
                  __m512d vdx0, vdx1, vdx2, vdx3;
                  __m512d vdy0, vdy1, vdy2, vdy3;
                  __m512d vd0, vd1, vd2, vd3;
/*
 *                NOTE: don't need those temp regs if no_sync used 
 */
                  __m512d vtfx0, vtfx1, vtfx2, vtfx3;
                  __m512d vtfy0, vtfy1, vtfy2, vtfy3;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();
                  
                  // init temp sum 
		  vtfx0 = _mm512_set1_pd(0.0); vtfy0 = _mm512_set1_pd(0.0);
		  vtfx1 = _mm512_set1_pd(0.0); vtfy1 = _mm512_set1_pd(0.0);
		  vtfx2 = _mm512_set1_pd(0.0); vtfy2 = _mm512_set1_pd(0.0);
		  vtfx3 = _mm512_set1_pd(0.0); vtfy3 = _mm512_set1_pd(0.0);
/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     double xj = blasX[j];
                     register __m512d vxj = _mm512_set1_pd(xj);
                     double yj = blasY[j];
                     register __m512d vyj = _mm512_set1_pd(yj);
		  
                     //dx0 = xj - x0;
		     vdx0 = _mm512_sub_pd(vxj, vx0);
		     vdx1 = _mm512_sub_pd(vxj, vx1);
		     vdx2 = _mm512_sub_pd(vxj, vx2);
		     vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                     // dy0 = yj - y0;
		     vdy0 = _mm512_sub_pd(vyj, vy0);
		     vdy1 = _mm512_sub_pd(vyj, vy1);
		     vdy2 = _mm512_sub_pd(vyj, vy2);
		     vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     vd0 = _mm512_mul_pd(vdx0, vdx0);
		     vd1 = _mm512_mul_pd(vdx1, vdx1);
		     vd2 = _mm512_mul_pd(vdx2, vdx2);
		     vd3 = _mm512_mul_pd(vdx3, vdx3);
                     
                     vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                     vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                     vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                     vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                     
                     vd0 = _MM512_RCP_PD(vd0); 
                     vd1 = _MM512_RCP_PD(vd1); 
                     vd2 = _MM512_RCP_PD(vd2); 
                     vd3 = _MM512_RCP_PD(vd3); 
		     
                     //tfx0 += dx0 * d0;
                     vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                     vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                     vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                     vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                     //tfy0 += dy0 * d0;
                     vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                     vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                     vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                     vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d: lower cleanup case = %d\n", i, 
                           i-nthreads*chunksize); 
         #endif
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
		  
                        //dx0 = xj - x0;
		        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        
                        vd0 = _MM512_RCP_PD(vd0); 
                        vd1 = _MM512_RCP_PD(vd1); 
                        vd2 = _MM512_RCP_PD(vd2); 
                        vd3 = _MM512_RCP_PD(vd3); 
                  
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                     }

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                     for (j = i; j < i+32; j++) 
                     {
                        unsigned int n = j-i;
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
                        register __m512d vzero = _mm512_set1_pd(0);
                        
                        // by default all 1 (for 8 elements)
                        unsigned int ik0, ik1, ik2, ik3; 
                        __mmask8 k0, k1, k2, k3; 
            #if 1
/*
 *                      Possible to further optimize.. will think later 
 */
                        ik0 = ( n >= 0 && n < 8 ) ? (255 & (~(1 << (n%8))))  : 255; 
                        ik1 = ( n >=8 && n < 8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik2 = ( n >=2*8 && n < 2*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik3 = ( n >=3*8 && n < 3*8+8 )? (255 & ~(1 << n%8))  : 255; 
		        
                        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        k0 = _cvtu32_mask8(ik0);
                        vd0 = _MM512_MASKZ_RCP_PD(k0, vd0); 
                     
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd1 = _MM512_RCP_PD(vd1); 
                        k1 = _cvtu32_mask8(ik1);
                        vd1 = _MM512_MASKZ_RCP_PD(k1, vd1); 
                     
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        k2 = _cvtu32_mask8(ik2);
                        vd2 = _MM512_MASKZ_RCP_PD(k2, vd2); 
                     
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        k3 = _cvtu32_mask8(ik3);
                        vd3 = _MM512_MASKZ_RCP_PD(k3, vd3); 
                  
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
            #endif
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - 32)/nthreads; 
         #if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "upper half case: chunksize = %d\n", 
                           chunksize);
                  }
         #endif

                  for(j = (i+32)+id*chunksize; j < (i+32)+(id+1)*chunksize; j++)
                  {
                     double xj = blasX[j];
                     register __m512d vxj = _mm512_set1_pd(xj);
                     double yj = blasY[j];
                     register __m512d vyj = _mm512_set1_pd(yj);
		  
                     //dx0 = xj - x0;
		     vdx0 = _mm512_sub_pd(vxj, vx0);
		     vdx1 = _mm512_sub_pd(vxj, vx1);
		     vdx2 = _mm512_sub_pd(vxj, vx2);
		     vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                     // dy0 = yj - y0;
		     vdy0 = _mm512_sub_pd(vyj, vy0);
		     vdy1 = _mm512_sub_pd(vyj, vy1);
		     vdy2 = _mm512_sub_pd(vyj, vy2);
		     vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     vd0 = _mm512_mul_pd(vdx0, vdx0);
		     vd1 = _mm512_mul_pd(vdx1, vdx1);
		     vd2 = _mm512_mul_pd(vdx2, vdx2);
		     vd3 = _mm512_mul_pd(vdx3, vdx3);
                     
                     vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                     vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                     vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                     vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                     
                     vd0 = _MM512_RCP_PD(vd0); 
                     vd1 = _MM512_RCP_PD(vd1); 
                     vd2 = _MM512_RCP_PD(vd2); 
                     vd3 = _MM512_RCP_PD(vd3); 
                  
                     //tfx0 += dx0 * d0;
                     vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                     vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                     vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                     vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                     //tfy0 += dy0 * d0;
                     vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                     vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                     vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                     vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d (id=%d): lower cleanup case\n", i, id); 
         #endif
                     for (j=(i+32)+nthreads*chunksize; j < M; j++)
                     {
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
		  
                        //dx0 = xj - x0;
		        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        
                        vd0 = _MM512_RCP_PD(vd0); 
                        vd1 = _MM512_RCP_PD(vd1); 
                        vd2 = _MM512_RCP_PD(vd2); 
                        vd3 = _MM512_RCP_PD(vd3); 
                  
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                     }
                  }
/*
 *                Reduction 
 */
                  #pragma omp critical
                  {
                     //fx0 += tfx0; 
		     vfx0 = _mm512_add_pd(vfx0, vtfx0);
		     vfx1 = _mm512_add_pd(vfx1, vtfx1);
		     vfx2 = _mm512_add_pd(vfx2, vtfx2);
		     vfx3 = _mm512_add_pd(vfx3, vtfx3);
                     
                     //fx1 += tfx1; 
		     vfy0 = _mm512_add_pd(vfy0, vtfy0);
		     vfy1 = _mm512_add_pd(vfy1, vtfy1);
		     vfy2 = _mm512_add_pd(vfy2, vtfy2);
		     vfy3 = _mm512_add_pd(vfy3, vtfy3);
                  }
               }
/*
 *             Out of parallel region 
 */
               _mm512_storeu_pd(pb_X+ind, vfx0); 
               _mm512_storeu_pd(pb_X+ind+8, vfx1); 
               _mm512_storeu_pd(pb_X+ind+2*8, vfx2); 
               _mm512_storeu_pd(pb_X+ind+3*8, vfx3); 
               
               _mm512_storeu_pd(pb_Y+ind, vfy0); 
               _mm512_storeu_pd(pb_Y+ind+8, vfy1); 
               _mm512_storeu_pd(pb_Y+ind+2*8, vfy2); 
               _mm512_storeu_pd(pb_Y+ind+3*8, vfy3); 
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

#elif defined (MDIM_VEC_UR8_NOSYNC)
   
   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();

/*
 *    NOTE: we have different version of reciprocals... I'm defining it here
 *                   NOTE: rcp14_pd inst... precision 2^-14
 *                   rcp28_pd has relative error 2^-28 ... rounding??? 
 *                   NOTE: reciprocal is highly error prone for over/underflow
 *                   FIXME: to compare apple to apple, make div version 
 */
      #define _MM512_RCP_PD _mm512_rcp14_pd  // rcp28 -- AVX512ER 
      #define _MM512_MASKZ_RCP_PD _mm512_maskz_rcp14_pd  // rcp28

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
	
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 64)
            {
	       int ind = i-b*BATCHSIZE;
               int m, n, k;
               register __m512d vx0, vx1, vx2, vx3, vx4, vx5, vx6, vx7;
               register __m512d vy0, vy1, vy2, vy3, vy4, vy5, vy6, vy7;
/*
 *             to minimize critical section, we are using extra memory here
 *                Total mem = NUMOFTHREADS * 1KB
 */
               double sum[NUMOFTHREADS][16][8]; // 8+8 vector 
	       for (m=0; m < NUMOFTHREADS; m++)
                 for (n=0; n < 16; n++)
                     for (k=0; k < 8; k++)
                        sum[m][n][k] = 0.0;

	       vx0 = _mm512_loadu_pd(blasX + i);
	       vx1 = _mm512_loadu_pd(blasX + i+8);
	       vx2 = _mm512_loadu_pd(blasX + i+2*8);
	       vx3 = _mm512_loadu_pd(blasX + i+3*8);
	       vx4 = _mm512_loadu_pd(blasX + i+4*8);
	       vx5 = _mm512_loadu_pd(blasX + i+5*8);
	       vx6 = _mm512_loadu_pd(blasX + i+6*8);
	       vx7 = _mm512_loadu_pd(blasX + i+7*8);

	       vy0 = _mm512_loadu_pd(blasY + i);
	       vy1 = _mm512_loadu_pd(blasY + i+1*8);
	       vy2 = _mm512_loadu_pd(blasY + i+2*8);
	       vy3 = _mm512_loadu_pd(blasY + i+3*8);
	       vy4 = _mm512_loadu_pd(blasY + i+4*8);
	       vy5 = _mm512_loadu_pd(blasY + i+5*8);
	       vy6 = _mm512_loadu_pd(blasY + i+6*8);
	       vy7 = _mm512_loadu_pd(blasY + i+7*8);
					
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j; 
                  __m512d vdx0, vdx1, vdx2, vdx3, vdx4, vdx5, vdx6, vdx7;
                  __m512d vdy0, vdy1, vdy2, vdy3, vdy4, vdy5, vdy6, vdy7;
                  __m512d vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7;
/*
 *                NOTE: don't need those temp regs if no_sync used 
 */
                  __m512d vtfx0, vtfx1, vtfx2, vtfx3, vtfx4, vtfx5, vtfx6, vtfx7;
                  __m512d vtfy0, vtfy1, vtfy2, vtfy3, vtfy4, vtfy5, vtfy6, vtfy7;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();
                  
                  // init temp sum 
		  vtfx0 = _mm512_set1_pd(0.0); vtfy0 = _mm512_set1_pd(0.0);
		  vtfx1 = _mm512_set1_pd(0.0); vtfy1 = _mm512_set1_pd(0.0);
		  vtfx2 = _mm512_set1_pd(0.0); vtfy2 = _mm512_set1_pd(0.0);
		  vtfx3 = _mm512_set1_pd(0.0); vtfy3 = _mm512_set1_pd(0.0);
		  vtfx4 = _mm512_set1_pd(0.0); vtfy4 = _mm512_set1_pd(0.0);
		  vtfx5 = _mm512_set1_pd(0.0); vtfy5 = _mm512_set1_pd(0.0);
		  vtfx6 = _mm512_set1_pd(0.0); vtfy6 = _mm512_set1_pd(0.0);
		  vtfx7 = _mm512_set1_pd(0.0); vtfy7 = _mm512_set1_pd(0.0);
/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     double xj = blasX[j];
                     register __m512d vxj = _mm512_set1_pd(xj);
                     double yj = blasY[j];
                     register __m512d vyj = _mm512_set1_pd(yj);
		  
                     //dx0 = xj - x0;
		     vdx0 = _mm512_sub_pd(vxj, vx0);
		     vdx1 = _mm512_sub_pd(vxj, vx1);
		     vdx2 = _mm512_sub_pd(vxj, vx2);
		     vdx3 = _mm512_sub_pd(vxj, vx3);
		     vdx4 = _mm512_sub_pd(vxj, vx4);
		     vdx5 = _mm512_sub_pd(vxj, vx5);
		     vdx6 = _mm512_sub_pd(vxj, vx6);
		     vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                     // dy0 = yj - y0;
		     vdy0 = _mm512_sub_pd(vyj, vy0);
		     vdy1 = _mm512_sub_pd(vyj, vy1);
		     vdy2 = _mm512_sub_pd(vyj, vy2);
		     vdy3 = _mm512_sub_pd(vyj, vy3);
		     vdy4 = _mm512_sub_pd(vyj, vy4);
		     vdy5 = _mm512_sub_pd(vyj, vy5);
		     vdy6 = _mm512_sub_pd(vyj, vy6);
		     vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     vd0 = _mm512_mul_pd(vdx0, vdx0);
                     vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                     vd0 = _MM512_RCP_PD(vd0); 
                     
		     vd1 = _mm512_mul_pd(vdx1, vdx1);
                     vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                     vd1 = _MM512_RCP_PD(vd1); 
                     
		     vd2 = _mm512_mul_pd(vdx2, vdx2);
                     vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                     vd2 = _MM512_RCP_PD(vd2); 
                     
		     vd3 = _mm512_mul_pd(vdx3, vdx3);
                     vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                     vd3 = _MM512_RCP_PD(vd3); 
                  
		     vd4 = _mm512_mul_pd(vdx4, vdx4);
                     vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                     vd4 = _MM512_RCP_PD(vd4); 
                     
                     
		     vd5 = _mm512_mul_pd(vdx5, vdx5);
                     vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                     vd5 = _MM512_RCP_PD(vd5); 
                     
                     
		     vd6 = _mm512_mul_pd(vdx6, vdx6);
                     vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                     vd6 = _MM512_RCP_PD(vd6); 
                     
		     vd7 = _mm512_mul_pd(vdx7, vdx7);
                     vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                     vd7 = _MM512_RCP_PD(vd7); 
		
		     
                     //tfx0 += dx0 * d0;
                     vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                     vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                     vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                     vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                     vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                     vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                     vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                     vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                     //tfy0 += dy0 * d0;
                     vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                     vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                     vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                     vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                     vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                     vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                     vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                     vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d: lower cleanup case = %d\n", i, 
                           i-nthreads*chunksize); 
         #endif
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
		  
                        //dx0 = xj - x0;
		        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
		        vdx4 = _mm512_sub_pd(vxj, vx4);
		        vdx5 = _mm512_sub_pd(vxj, vx5);
		        vdx6 = _mm512_sub_pd(vxj, vx6);
		        vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
		        vdy4 = _mm512_sub_pd(vyj, vy4);
		        vdy5 = _mm512_sub_pd(vyj, vy5);
		        vdy6 = _mm512_sub_pd(vyj, vy6);
		        vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        vd0 = _MM512_RCP_PD(vd0); 
                     
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd1 = _MM512_RCP_PD(vd1); 
                     
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        vd2 = _MM512_RCP_PD(vd2); 
                     
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        vd3 = _MM512_RCP_PD(vd3); 
                  
		        vd4 = _mm512_mul_pd(vdx4, vdx4);
                        vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                        vd4 = _MM512_RCP_PD(vd4); 
                     
		        vd5 = _mm512_mul_pd(vdx5, vdx5);
                        vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                        vd5 = _MM512_RCP_PD(vd5); 
                     
		        vd6 = _mm512_mul_pd(vdx6, vdx6);
                        vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                        vd6 = _MM512_RCP_PD(vd6); 
                     
		        vd7 = _mm512_mul_pd(vdx7, vdx7);
                        vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                        vd7 = _MM512_RCP_PD(vd7); 
		     
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                        vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                        vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                        vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                        vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                        vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                        vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                        vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                        vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
                     }

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 *             Hopefully compiler will unroll and get rid of the conditions
 */
                     for (j = i; j < i+64; j++) 
                     {
                        unsigned int n = j-i;
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
                        register __m512d vzero = _mm512_set1_pd(0);
                        
                        // by default all 1 (for 8 elements)
                        unsigned int ik0, ik1, ik2, ik3, ik4, ik5, ik6, ik7; 
                        __mmask8 k0, k1, k2, k3, k4, k5, k6, k7; 
            #if 1
/*
 *                      Possible to further optimize.. will think later 
 */
                        ik0 = ( n >= 0 && n < 8 ) ? (255 & (~(1 << (n%8))))  : 255; 
                        ik1 = ( n >=8 && n < 8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik2 = ( n >=2*8 && n < 2*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik3 = ( n >=3*8 && n < 3*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik4 = ( n >=4*8 && n < 4*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik5 = ( n >=5*8 && n < 5*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik6 = ( n >=6*8 && n < 6*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik7 = ( n >=7*8 && n < 7*8+8 )? (255 & ~(1 << n%8))  : 255; 
		        
                        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
		        vdx4 = _mm512_sub_pd(vxj, vx4);
		        vdx5 = _mm512_sub_pd(vxj, vx5);
		        vdx6 = _mm512_sub_pd(vxj, vx6);
		        vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
		        vdy4 = _mm512_sub_pd(vyj, vy4);
		        vdy5 = _mm512_sub_pd(vyj, vy5);
		        vdy6 = _mm512_sub_pd(vyj, vy6);
		        vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        k0 = _cvtu32_mask8(ik0);
                        //vd0 = _MM512_MASKZ_RCP_PD(vd0, k0, vd0); 
                        vd0 = _MM512_MASKZ_RCP_PD(k0, vd0); 
                     
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd1 = _MM512_RCP_PD(vd1); 
                        k1 = _cvtu32_mask8(ik1);
                        //vd1 = _MM512_MASKZ_RCP_PD(vd1, k1, vd1); 
                        vd1 = _MM512_MASKZ_RCP_PD(k1, vd1); 
                     
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        k2 = _cvtu32_mask8(ik2);
                        //vd2 = _MM512_MASKZ_RCP_PD(vd2, k2, vd2); 
                        vd2 = _MM512_MASKZ_RCP_PD(k2, vd2); 
                     
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        k3 = _cvtu32_mask8(ik3);
                        //vd3 = _MM512_MASKZ_RCP_PD(vd3, k3, vd3); 
                        vd3 = _MM512_MASKZ_RCP_PD(k3, vd3); 
                  
		        vd4 = _mm512_mul_pd(vdx4, vdx4);
                        vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                        k4 = _cvtu32_mask8(ik4);
                        //vd4 = _MM512_MASKZ_RCP_PD(vd4, k4, vd4); 
                        vd4 = _MM512_MASKZ_RCP_PD(k4, vd4); 
                     
		        vd5 = _mm512_mul_pd(vdx5, vdx5);
                        vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                        k5 = _cvtu32_mask8(ik5);
                        //vd5 = _MM512_MASKZ_RCP_PD(vd5, k5, vd5); 
                        vd5 = _MM512_MASKZ_RCP_PD(k5, vd5); 
                     
		        vd6 = _mm512_mul_pd(vdx6, vdx6);
                        vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                        k6 = _cvtu32_mask8(ik6);
                        //vd6 = _MM512_MASKZ_RCP_PD(vd6, k6, vd6); 
                        vd6 = _MM512_MASKZ_RCP_PD(k6, vd6); 
                     
		        vd7 = _mm512_mul_pd(vdx7, vdx7);
                        vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                        k7 = _cvtu32_mask8(ik7);
                        //vd7 = _MM512_MASKZ_RCP_PD(vd7, k7, vd7); 
                        vd7 = _MM512_MASKZ_RCP_PD(k7, vd7); 
		     
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                        vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                        vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                        vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                        vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                        vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                        vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                        vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                        vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
            #endif
                     }
                  }
/*
 *                Upper part  
 *
                  for(INDEXTYPE j = i+8; j < graph.rows; j += 1)
 */ 
                  chunksize = (M - i - 64)/nthreads; 
         #if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "upper half case: chunksize = %d\n", 
                           chunksize);
                  }
         #endif

                  for(j = (i+64)+id*chunksize; j < (i+64)+(id+1)*chunksize; j++)
                  {
                     double xj = blasX[j];
                     register __m512d vxj = _mm512_set1_pd(xj);
                     double yj = blasY[j];
                     register __m512d vyj = _mm512_set1_pd(yj);
		  
                     //dx0 = xj - x0;
		     vdx0 = _mm512_sub_pd(vxj, vx0);
		     vdx1 = _mm512_sub_pd(vxj, vx1);
		     vdx2 = _mm512_sub_pd(vxj, vx2);
		     vdx3 = _mm512_sub_pd(vxj, vx3);
		     vdx4 = _mm512_sub_pd(vxj, vx4);
		     vdx5 = _mm512_sub_pd(vxj, vx5);
		     vdx6 = _mm512_sub_pd(vxj, vx6);
		     vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                     // dy0 = yj - y0;
		     vdy0 = _mm512_sub_pd(vyj, vy0);
		     vdy1 = _mm512_sub_pd(vyj, vy1);
		     vdy2 = _mm512_sub_pd(vyj, vy2);
		     vdy3 = _mm512_sub_pd(vyj, vy3);
		     vdy4 = _mm512_sub_pd(vyj, vy4);
		     vdy5 = _mm512_sub_pd(vyj, vy5);
		     vdy6 = _mm512_sub_pd(vyj, vy6);
		     vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     vd0 = _mm512_mul_pd(vdx0, vdx0);
                     vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                     vd0 = _MM512_RCP_PD(vd0); 
                     
		     vd1 = _mm512_mul_pd(vdx1, vdx1);
                     vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                     vd1 = _MM512_RCP_PD(vd1); 
                     
		     vd2 = _mm512_mul_pd(vdx2, vdx2);
                     vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                     vd2 = _MM512_RCP_PD(vd2); 
                     
		     vd3 = _mm512_mul_pd(vdx3, vdx3);
                     vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                     vd3 = _MM512_RCP_PD(vd3); 
                  
		     vd4 = _mm512_mul_pd(vdx4, vdx4);
                     vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                     vd4 = _MM512_RCP_PD(vd4); 
                     
                     
		     vd5 = _mm512_mul_pd(vdx5, vdx5);
                     vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                     vd5 = _MM512_RCP_PD(vd5); 
                     
                     
		     vd6 = _mm512_mul_pd(vdx6, vdx6);
                     vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                     vd6 = _MM512_RCP_PD(vd6); 
                     
		     vd7 = _mm512_mul_pd(vdx7, vdx7);
                     vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                     vd7 = _MM512_RCP_PD(vd7); 
		
		     
                     //tfx0 += dx0 * d0;
                     vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                     vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                     vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                     vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                     vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                     vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                     vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                     vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                     //tfy0 += dy0 * d0;
                     vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                     vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                     vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                     vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                     vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                     vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                     vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                     vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d (id=%d): lower cleanup case\n", i, id); 
         #endif
                     for (j=(i+64)+nthreads*chunksize; j < M; j++)
                     {
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
		  
                        //dx0 = xj - x0;
		        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
		        vdx4 = _mm512_sub_pd(vxj, vx4);
		        vdx5 = _mm512_sub_pd(vxj, vx5);
		        vdx6 = _mm512_sub_pd(vxj, vx6);
		        vdx7 = _mm512_sub_pd(vxj, vx7);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
		        vdy4 = _mm512_sub_pd(vyj, vy4);
		        vdy5 = _mm512_sub_pd(vyj, vy5);
		        vdy6 = _mm512_sub_pd(vyj, vy6);
		        vdy7 = _mm512_sub_pd(vyj, vy7);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        vd0 = _MM512_RCP_PD(vd0); 
                     
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd1 = _MM512_RCP_PD(vd1); 
                     
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        vd2 = _MM512_RCP_PD(vd2); 
                     
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        vd3 = _MM512_RCP_PD(vd3); 
                  
		        vd4 = _mm512_mul_pd(vdx4, vdx4);
                        vd4 = _mm512_fmadd_pd(vdy4, vdy4, vdx4);
                        vd4 = _MM512_RCP_PD(vd4); 
                     
                     
		        vd5 = _mm512_mul_pd(vdx5, vdx5);
                        vd5 = _mm512_fmadd_pd(vdy5, vdy5, vdx5);
                        vd5 = _MM512_RCP_PD(vd5); 
                     
                     
		        vd6 = _mm512_mul_pd(vdx6, vdx6);
                        vd6 = _mm512_fmadd_pd(vdy6, vdy6, vdx6);
                        vd6 = _MM512_RCP_PD(vd6); 
                     
		        vd7 = _mm512_mul_pd(vdx7, vdx7);
                        vd7 = _mm512_fmadd_pd(vdy7, vdy7, vdx7);
                        vd7 = _MM512_RCP_PD(vd7); 
		
		     
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);
                        vtfx4 = _mm512_fmadd_pd(vdx4, vd4, vtfx4);
                        vtfx5 = _mm512_fmadd_pd(vdx5, vd5, vtfx5);
                        vtfx6 = _mm512_fmadd_pd(vdx6, vd6, vtfx6);
                        vtfx7 = _mm512_fmadd_pd(vdx7, vd7, vtfx7);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                        vtfy4 = _mm512_fmadd_pd(vdy4, vd4, vtfy4);
                        vtfy5 = _mm512_fmadd_pd(vdy5, vd5, vtfy5);
                        vtfy6 = _mm512_fmadd_pd(vdy6, vd6, vtfy6);
                        vtfy7 = _mm512_fmadd_pd(vdy7, vd7, vtfy7);
                     }
                  }
/*
 *                Reduction without synchronization  
 */
                  {
                     register __m512d tv; 
                     
                     tv = _mm512_loadu_pd(sum[id][0]);
                     vtfx0 = _mm512_add_pd(vtfx0, tv);
                     _mm512_storeu_pd(sum[id][0], vtfx0);
                     
                     tv = _mm512_loadu_pd(sum[id][1]);
                     vtfx1 = _mm512_add_pd(vtfx0, tv);
                     _mm512_storeu_pd(sum[id][1], vtfx1);
                     
                     tv = _mm512_loadu_pd(sum[id][2]);
                     vtfx2 = _mm512_add_pd(vtfx2, tv);
                     _mm512_storeu_pd(sum[id][2], vtfx2);

                     tv = _mm512_loadu_pd(sum[id][3]);
                     vtfx3 = _mm512_add_pd(vtfx3, tv);
                     _mm512_storeu_pd(sum[id][3], vtfx3);
                     
                     tv = _mm512_loadu_pd(sum[id][4]);
                     vtfx4 = _mm512_add_pd(vtfx4, tv);
                     _mm512_storeu_pd(sum[id][4], vtfx4);
                     
                     tv = _mm512_loadu_pd(sum[id][5]);
                     vtfx5 = _mm512_add_pd(vtfx5, tv);
                     _mm512_storeu_pd(sum[id][5], vtfx5);

                     tv = _mm512_loadu_pd(sum[id][6]);
                     vtfx6 = _mm512_add_pd(vtfx6, tv);
                     _mm512_storeu_pd(sum[id][6], vtfx6);

                     tv = _mm512_loadu_pd(sum[id][7]);
                     vtfx7 = _mm512_add_pd(vtfx7, tv);
                     _mm512_storeu_pd(sum[id][7], vtfx7);
                     
                     //fy1 += tfy1; 
                     tv = _mm512_loadu_pd(sum[id][0+8]);
                     vtfy0 = _mm512_add_pd(vtfy0, tv);
                     _mm512_storeu_pd(sum[id][0+8], vtfy0);
                     
                     tv = _mm512_loadu_pd(sum[id][1+8]);
                     vtfy1 = _mm512_add_pd(vtfy0, tv);
                     _mm512_storeu_pd(sum[id][1+8], vtfy1);
                     
                     tv = _mm512_loadu_pd(sum[id][2+8]);
                     vtfy2 = _mm512_add_pd(vtfy2, tv);
                     _mm512_storeu_pd(sum[id][2+8], vtfy2);

                     tv = _mm512_loadu_pd(sum[id][3+8]);
                     vtfy3 = _mm512_add_pd(vtfy3, tv);
                     _mm512_storeu_pd(sum[id][3+8], vtfy3);
                     
                     tv = _mm512_loadu_pd(sum[id][4+8]);
                     vtfy4 = _mm512_add_pd(vtfy4, tv);
                     _mm512_storeu_pd(sum[id][4+8], vtfy4);
                     
                     tv = _mm512_loadu_pd(sum[id][5+8]);
                     vtfy5 = _mm512_add_pd(vtfy5, tv);
                     _mm512_storeu_pd(sum[id][5+8], vtfy5);

                     tv = _mm512_loadu_pd(sum[id][6+8]);
                     vtfy6 = _mm512_add_pd(vtfy6, tv);
                     _mm512_storeu_pd(sum[id][6+8], vtfy6);

                     tv = _mm512_loadu_pd(sum[id][7+8]);
                     vtfy7 = _mm512_add_pd(vtfy7, tv);
                     _mm512_storeu_pd(sum[id][7+8], vtfy7);
                  }
               }
/*
 *             Out of parallel region 
 */
/*
 *             sum the results up:  
 */
               for (m=1; m < NUMOFTHREADS; m++)
                  for (n=0; n < 16; n++)
                     for (k=0; k < 8; k++)
                        sum[0][n][k] += sum[m][n][k];

               __m512d tvsum;

               tvsum = _mm512_loadu_pd(sum[0][0]);
               _mm512_storeu_pd(pb_X+ind, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][1]);
               _mm512_storeu_pd(pb_X+ind+8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][2]);
               _mm512_storeu_pd(pb_X+ind+2*8, tvsum);

               tvsum = _mm512_loadu_pd(sum[0][3]);
               _mm512_storeu_pd(pb_X+ind+3*8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][4]);
               _mm512_storeu_pd(pb_X+ind+4*8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][5]);
               _mm512_storeu_pd(pb_X+ind+5*8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][6]);
               _mm512_storeu_pd(pb_X+ind+6*8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][7]);
               _mm512_storeu_pd(pb_X+ind+7*8, tvsum); 
               
               // Y  
               tvsum = _mm512_loadu_pd(sum[0][8+0]);
               _mm512_storeu_pd(pb_Y+ind, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][8+1]);
               _mm512_storeu_pd(pb_Y+ind+8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][8+2]);
               _mm512_storeu_pd(pb_Y+ind+2*8, tvsum);

               tvsum = _mm512_loadu_pd(sum[0][8+3]);
               _mm512_storeu_pd(pb_Y+ind+3*8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][8+4]);
               _mm512_storeu_pd(pb_Y+ind+4*8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][8+5]);
               _mm512_storeu_pd(pb_Y+ind+5*8, tvsum);

               tvsum = _mm512_loadu_pd(sum[0][8+6]);
               _mm512_storeu_pd(pb_Y+ind+6*8, tvsum);

               tvsum = _mm512_loadu_pd(sum[0][8+7]);
               _mm512_storeu_pd(pb_Y+ind+7*8, tvsum); 
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

#elif defined(MDIM_VEC_UR4_NOSYNC)

   vector<VALUETYPE> newalgo::EfficientVersionMdim(INDEXTYPE ITERATIONS, 
         INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE)
   {
      INDEXTYPE LOOP = 0;
      VALUETYPE start, end, ENERGY, ENERGY0, *pb_X, *pb_Y;
      VALUETYPE STEP = 1.0;
      
      
      vector<VALUETYPE> result;
      pb_X = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      pb_Y = static_cast<VALUETYPE *> (::operator new (sizeof(VALUETYPE[BATCHSIZE])));
      ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();

/*
 *    NOTE: we have different version of reciprocals... I'm defining it here
 *                   NOTE: rcp14_pd inst... precision 2^-14
 *                   rcp28_pd has relative error 2^-28 ... rounding??? 
 *                   NOTE: reciprocal is highly error prone for over/underflow
 *                   FIXME: to compare apple to apple, make div version 
 */
      #define _MM512_RCP_PD _mm512_rcp14_pd  // rcp28 -- AVX512ER 
      #define _MM512_MASKZ_RCP_PD _mm512_maskz_rcp14_pd  // rcp28

      omp_set_num_threads(NUMOFTHREADS);
/*
 *    time included initDFS 
 */
      start = omp_get_wtime();

      initDFS();
      
      while(LOOP < ITERATIONS)
      {
         ENERGY0 = ENERGY;
         ENERGY = 0;
         for(int i = 0; i < BATCHSIZE; i++)
         {
            pb_X[i] = pb_Y[i] = 0;
         }
         
	 for(INDEXTYPE b = 0; b < (graph.rows / BATCHSIZE); b += 1)
         {
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 32)
            {
	       int ind = i-b*BATCHSIZE;
               int m, n, k;
               register __m512d vx0, vx1, vx2, vx3;
               register __m512d vy0, vy1, vy2, vy3;
/*
 *             to minimize critical section, we are using extra memory here
 *                Total mem = NUMOFTHREADS * 0.5KB
 */
               double sum[NUMOFTHREADS][4+4][8]; // 4+4 vector 
	       for (m=0; m < NUMOFTHREADS; m++)
                 for (n=0; n < 8; n++)
                     for (k=0; k < 8; k++)
                        sum[m][n][k] = 0.0;

	       vx0 = _mm512_loadu_pd(blasX + i);
	       vx1 = _mm512_loadu_pd(blasX + i+8);
	       vx2 = _mm512_loadu_pd(blasX + i+2*8);
	       vx3 = _mm512_loadu_pd(blasX + i+3*8);

	       vy0 = _mm512_loadu_pd(blasY + i);
	       vy1 = _mm512_loadu_pd(blasY + i+1*8);
	       vy2 = _mm512_loadu_pd(blasY + i+2*8);
	       vy3 = _mm512_loadu_pd(blasY + i+3*8);
					
               #pragma omp parallel
               {
                  int id, nthreads; 
                  int chunksize;
                  int j; 
                  __m512d vdx0, vdx1, vdx2, vdx3;
                  __m512d vdy0, vdy1, vdy2, vdy3;
                  __m512d vd0, vd1, vd2, vd3;
                  
                  __m512d vtfx0, vtfx1, vtfx2, vtfx3;
                  __m512d vtfy0, vtfy1, vtfy2, vtfy3;

                  const int M = graph.rows; 
                  id = omp_get_thread_num();
                  nthreads = omp_get_num_threads();
                  
                  // init temp sum 
		  vtfx0 = _mm512_set1_pd(0.0); vtfy0 = _mm512_set1_pd(0.0);
		  vtfx1 = _mm512_set1_pd(0.0); vtfy1 = _mm512_set1_pd(0.0);
		  vtfx2 = _mm512_set1_pd(0.0); vtfy2 = _mm512_set1_pd(0.0);
		  vtfx3 = _mm512_set1_pd(0.0); vtfy3 = _mm512_set1_pd(0.0);
/*
 *                j is up to i... lower up case  
 */
                  chunksize = i/nthreads; 
                  
                  for(j = id*chunksize; j < (id+1)*chunksize; j++)
                  {
                     double xj = blasX[j];
                     register __m512d vxj = _mm512_set1_pd(xj);
                     double yj = blasY[j];
                     register __m512d vyj = _mm512_set1_pd(yj);
		  
                     //dx0 = xj - x0;
		     vdx0 = _mm512_sub_pd(vxj, vx0);
		     vdx1 = _mm512_sub_pd(vxj, vx1);
		     vdx2 = _mm512_sub_pd(vxj, vx2);
		     vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                     // dy0 = yj - y0;
		     vdy0 = _mm512_sub_pd(vyj, vy0);
		     vdy1 = _mm512_sub_pd(vyj, vy1);
		     vdy2 = _mm512_sub_pd(vyj, vy2);
		     vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     vd0 = _mm512_mul_pd(vdx0, vdx0);
		     vd1 = _mm512_mul_pd(vdx1, vdx1);
		     vd2 = _mm512_mul_pd(vdx2, vdx2);
		     vd3 = _mm512_mul_pd(vdx3, vdx3);
                     
                     vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                     vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                     vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                     vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                     
                     vd0 = _MM512_RCP_PD(vd0); 
                     vd1 = _MM512_RCP_PD(vd1); 
                     vd2 = _MM512_RCP_PD(vd2); 
                     vd3 = _MM512_RCP_PD(vd3); 
                  
                     //tfx0 += dx0 * d0;
                     vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                     vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                     vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                     vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                     //tfy0 += dy0 * d0;
                     vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                     vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                     vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                     vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                  }
/*
 *                cleanup of lower  
 */
                  #pragma omp single nowait
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d: lower cleanup case = %d\n", i, 
                           i-nthreads*chunksize); 
         #endif
                     for (j=nthreads*chunksize; j < i; j++)
                     {
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
		  
                        //dx0 = xj - x0;
		        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        
                        vd0 = _MM512_RCP_PD(vd0); 
                        vd1 = _MM512_RCP_PD(vd1); 
                        vd2 = _MM512_RCP_PD(vd2); 
                        vd3 = _MM512_RCP_PD(vd3); 
                  
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                     }

/*
 *             location where we need to skip some points i==j
 *             NOTE: its UR iteration, no need to optimize 
 */
                     for (j = i; j < i+32; j++) 
                     {
                        unsigned int n = j-i;
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
                        register __m512d vzero = _mm512_set1_pd(0);
                        
                        // by default all 1 (for 8 elements)
                        unsigned int ik0, ik1, ik2, ik3; 
                        __mmask8 k0, k1, k2, k3; 
            #if 1  // to debug 
/*
 *                      Possible to further optimize.. will think later 
 */
                        ik0 = ( n >= 0 && n < 8 ) ? (255 & (~(1 << (n%8))))  : 255; 
                        ik1 = ( n >=8 && n < 8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik2 = ( n >=2*8 && n < 2*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        ik3 = ( n >=3*8 && n < 3*8+8 )? (255 & ~(1 << n%8))  : 255; 
                        
		        
                        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        k0 = _cvtu32_mask8(ik0);
                        //vd0 = _MM512_MASKZ_RCP_PD(vd0, k0, vd0); 
                        vd0 = _MM512_MASKZ_RCP_PD(k0, vd0); 
                     
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd1 = _MM512_RCP_PD(vd1); 
                        k1 = _cvtu32_mask8(ik1);
                        //vd1 = _MM512_MASKZ_RCP_PD(vd1, k1, vd1); 
                        vd1 = _MM512_MASKZ_RCP_PD(k1, vd1); 
                     
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        k2 = _cvtu32_mask8(ik2);
                        //vd2 = _MM512_MASKZ_RCP_PD(vd2, k2, vd2); 
                        vd2 = _MM512_MASKZ_RCP_PD(k2, vd2); 
                     
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        k3 = _cvtu32_mask8(ik3);
                        //vd3 = _MM512_MASKZ_RCP_PD(vd3, k3, vd3); 
                        vd3 = _MM512_MASKZ_RCP_PD(k3, vd3); 
		     
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
            #endif
                     }
                  }
/*
 *                Upper part  
 *
 */ 
                  chunksize = (M - i - 32)/nthreads; 
         #if DEBUG == 1
                  if (id == 0)
                  {
                     fprintf(DOUT, "upper half case: chunksize = %d\n", 
                           chunksize);
                  }
         #endif

                  for(j = (i+32)+id*chunksize; j < (i+32)+(id+1)*chunksize; j++)
                  {
                     double xj = blasX[j];
                     register __m512d vxj = _mm512_set1_pd(xj);
                     double yj = blasY[j];
                     register __m512d vyj = _mm512_set1_pd(yj);
		  
                     //dx0 = xj - x0;
		     vdx0 = _mm512_sub_pd(vxj, vx0);
		     vdx1 = _mm512_sub_pd(vxj, vx1);
		     vdx2 = _mm512_sub_pd(vxj, vx2);
		     vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                     // dy0 = yj - y0;
		     vdy0 = _mm512_sub_pd(vyj, vy0);
		     vdy1 = _mm512_sub_pd(vyj, vy1);
		     vdy2 = _mm512_sub_pd(vyj, vy2);
		     vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                     //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		     vd0 = _mm512_mul_pd(vdx0, vdx0);
		     vd1 = _mm512_mul_pd(vdx1, vdx1);
		     vd2 = _mm512_mul_pd(vdx2, vdx2);
		     vd3 = _mm512_mul_pd(vdx3, vdx3);
                     
                     vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                     vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                     vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                     vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                     
                     vd0 = _MM512_RCP_PD(vd0); 
                     vd1 = _MM512_RCP_PD(vd1); 
                     vd2 = _MM512_RCP_PD(vd2); 
                     vd3 = _MM512_RCP_PD(vd3); 
                  
                     //tfx0 += dx0 * d0;
                     vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                     vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                     vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                     vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                     //tfy0 += dy0 * d0;
                     vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                     vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                     vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                     vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                  }
/*
 *                cleanup 
 */
                  #pragma omp single nowait 
                  {
         #if DEBUG == 1
                     fprintf(DOUT, "i=%d (id=%d): lower cleanup case\n", i, id); 
         #endif
                     for (j=(i+32)+nthreads*chunksize; j < M; j++)
                     {
                        double xj = blasX[j];
                        register __m512d vxj = _mm512_set1_pd(xj);
                        double yj = blasY[j];
                        register __m512d vyj = _mm512_set1_pd(yj);
		  
                        //dx0 = xj - x0;
		        vdx0 = _mm512_sub_pd(vxj, vx0);
		        vdx1 = _mm512_sub_pd(vxj, vx1);
		        vdx2 = _mm512_sub_pd(vxj, vx2);
		        vdx3 = _mm512_sub_pd(vxj, vx3);
	          
                        // dy0 = yj - y0;
		        vdy0 = _mm512_sub_pd(vyj, vy0);
		        vdy1 = _mm512_sub_pd(vyj, vy1);
		        vdy2 = _mm512_sub_pd(vyj, vy2);
		        vdy3 = _mm512_sub_pd(vyj, vy3);
	             
                        //d0 = 1.0 / (dx0 * dx0 + dy0 * dy0);
		        vd0 = _mm512_mul_pd(vdx0, vdx0);
                        vd0 = _mm512_fmadd_pd(vdy0, vdy0, vdx0);
                        vd0 = _MM512_RCP_PD(vd0); 
                     
		        vd1 = _mm512_mul_pd(vdx1, vdx1);
                        vd1 = _mm512_fmadd_pd(vdy1, vdy1, vdx1);
                        vd1 = _MM512_RCP_PD(vd1); 
                     
		        vd2 = _mm512_mul_pd(vdx2, vdx2);
                        vd2 = _mm512_fmadd_pd(vdy2, vdy2, vdx2);
                        vd2 = _MM512_RCP_PD(vd2); 
                     
		        vd3 = _mm512_mul_pd(vdx3, vdx3);
                        vd3 = _mm512_fmadd_pd(vdy3, vdy3, vdx3);
                        vd3 = _MM512_RCP_PD(vd3); 
                  
		     
                        //tfx0 += dx0 * d0;
                        vtfx0 = _mm512_fmadd_pd(vdx0, vd0, vtfx0);
                        vtfx1 = _mm512_fmadd_pd(vdx1, vd1, vtfx1);
                        vtfx2 = _mm512_fmadd_pd(vdx2, vd2, vtfx2);
                        vtfx3 = _mm512_fmadd_pd(vdx3, vd3, vtfx3);

                        //tfy0 += dy0 * d0;
                        vtfy0 = _mm512_fmadd_pd(vdy0, vd0, vtfy0);
                        vtfy1 = _mm512_fmadd_pd(vdy1, vd1, vtfy1);
                        vtfy2 = _mm512_fmadd_pd(vdy2, vd2, vtfy2);
                        vtfy3 = _mm512_fmadd_pd(vdy3, vd3, vtfy3);
                     }
                  }
/*
 *                Reduction without synchronization  
 */
                  {
                     register __m512d tv; 
                     
                     tv = _mm512_loadu_pd(sum[id][0]);
                     vtfx0 = _mm512_add_pd(vtfx0, tv);
                     _mm512_storeu_pd(sum[id][0], vtfx0);
                     
                     tv = _mm512_loadu_pd(sum[id][1]);
                     vtfx1 = _mm512_add_pd(vtfx0, tv);
                     _mm512_storeu_pd(sum[id][1], vtfx1);
                     
                     tv = _mm512_loadu_pd(sum[id][2]);
                     vtfx2 = _mm512_add_pd(vtfx2, tv);
                     _mm512_storeu_pd(sum[id][2], vtfx2);

                     tv = _mm512_loadu_pd(sum[id][3]);
                     vtfx3 = _mm512_add_pd(vtfx3, tv);
                     _mm512_storeu_pd(sum[id][3], vtfx3);
                     
                     //fy1 += tfy1; 
                     tv = _mm512_loadu_pd(sum[id][0+4]);
                     vtfy0 = _mm512_add_pd(vtfy0, tv);
                     _mm512_storeu_pd(sum[id][0+4], vtfy0);
                     
                     tv = _mm512_loadu_pd(sum[id][1+4]);
                     vtfy1 = _mm512_add_pd(vtfy0, tv);
                     _mm512_storeu_pd(sum[id][1+4], vtfy1);
                     
                     tv = _mm512_loadu_pd(sum[id][2+4]);
                     vtfy2 = _mm512_add_pd(vtfy2, tv);
                     _mm512_storeu_pd(sum[id][2+4], vtfy2);

                     tv = _mm512_loadu_pd(sum[id][3+4]);
                     vtfy3 = _mm512_add_pd(vtfy3, tv);
                     _mm512_storeu_pd(sum[id][3+4], vtfy3);
                     
                  }
               }
/*
 *             Out of parallel region 
 */
/*
 *             sum the results up:  
 */
               for (m=1; m < NUMOFTHREADS; m++)
                  for (n=0; n < 8; n++)
                     for (k=0; k < 8; k++)
                        sum[0][n][k] += sum[m][n][k];

               __m512d tvsum;

               tvsum = _mm512_loadu_pd(sum[0][0]);
               _mm512_storeu_pd(pb_X+ind, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][1]);
               _mm512_storeu_pd(pb_X+ind+8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][2]);
               _mm512_storeu_pd(pb_X+ind+2*8, tvsum);

               tvsum = _mm512_loadu_pd(sum[0][3]);
               _mm512_storeu_pd(pb_X+ind+3*8, tvsum); 
               
               // Y  
               tvsum = _mm512_loadu_pd(sum[0][4+0]);
               _mm512_storeu_pd(pb_Y+ind, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][4+1]);
               _mm512_storeu_pd(pb_Y+ind+8, tvsum); 
               
               tvsum = _mm512_loadu_pd(sum[0][4+2]);
               _mm512_storeu_pd(pb_Y+ind+2*8, tvsum);

               tvsum = _mm512_loadu_pd(sum[0][4+3]);
               _mm512_storeu_pd(pb_Y+ind+3*8, tvsum); 
               
            }
            // connected nodes 
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {   
               int ind = i-b*BATCHSIZE;
               VALUETYPE pbX=0.0, pbY=0.0;
               VALUETYPE dist, distX, distY;
               for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1)
               {
                  int v = graph.colids[j];
                  distX = blasX[v] - blasX[i];
                  distY = blasY[v] - blasY[i];
                  dist = (distX * distX + distY * distY);
                  dist = sqrt(dist) + 1.0 / dist;
                  pbX += distX * dist;
                  pbY += distY * dist;
               }
               pb_X[ind] = pbX - pb_X[ind];
               pb_Y[ind] = pbY - pb_Y[ind];
            }
	    /*printf("After:\n");
            for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++)
            {
               printf("%d = %lf,", i, pb_X[i-b*BATCHSIZE]);
            }
	    */
	    for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1)
            {
               int ind = i-b*BATCHSIZE;
               VALUETYPE d;
	       d = (pb_X[ind] * pb_X[ind] + pb_Y[ind] * pb_Y[ind]);
	       ENERGY += d ;
	       
               d = STEP / sqrt(d);
	       blasX[i] += pb_X[ind] * d;
               blasY[i] += pb_Y[ind] * d;
	       
               pb_X[ind] = 0;
	       pb_Y[ind] = 0;    
            }
         }
/*
 *    cleanup 
 */
			INDEXTYPE cleanup = (graph.rows/BATCHSIZE) * BATCHSIZE;
			#pragma omp parallel for schedule(dynamic) 
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                INDEXTYPE ind = i- cleanup;
                                VALUETYPE fx = 0, fy = 0, distX, distY, dist, dist2;
                                for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
                                        int v = graph.colids[j];
                                        distX = blasX[v] - blasX[i];
                                        distY = blasY[v] - blasY[i];
                                        dist = (distX * distX + distY * distY);
                                        dist = sqrt(dist) + 1.0 / dist;
                                        pb_X[ind] += distX * dist;
                                        pb_Y[ind] += distY * dist;
                                }
                                for(INDEXTYPE j = 0; j < i; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
                                        distX = blasX[j] - blasX[i];
                                        distY = blasY[j] - blasY[i];
                                        dist2 = 1.0 / (distX * distX + distY * distY);
                                        fx += distX * dist2;
                                        fy += distY * dist2;
                                }
                                pb_X[ind] = pb_X[ind] - fx;
                                pb_Y[ind] = pb_Y[ind] - fy;
                        }	
			for(INDEXTYPE i = cleanup; i < graph.rows; i += 1){
                                int ind = i-cleanup;
                                double dist = (pb_X[ind]*pb_X[ind] + pb_Y[ind]*pb_Y[ind]);
                                ENERGY += dist;
				dist = STEP / sqrt(dist);
				blasX[i] += pb_X[ind] * dist;
                                blasY[i] += pb_Y[ind] * dist;
				pb_X[ind] = pb_Y[ind] = 0;
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
         #if 0
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
         #endif
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }

#endif




