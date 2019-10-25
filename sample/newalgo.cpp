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
                cout << "Efficient Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficient Minbatch Energy:" << ENERGY << endl;
                cout << "Efficient Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("Efficient"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
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
                cout << "Efficientunroll Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Efficientunroll Minbatch Energy:" << ENERGY << endl;
                cout << "Efficientunroll Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFileEFF("EFFUR"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }
				
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
