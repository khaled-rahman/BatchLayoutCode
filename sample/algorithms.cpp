#include "algorithms.h"

	algorithms::algorithms(CSR<INDEXTYPE, VALUETYPE> &A_csr, string input, string outputd, int init, double weight, double th, string ifile){
		graph.make_empty();
		graph = A_csr;
		outputdir = outputd;
		initfile = ifile;
		//cout << initfile << endl;
		W = weight;
		filename = input;
		threshold = th;
		nCoordinates = static_cast<Coordinate<VALUETYPE> *> (::operator new (sizeof(Coordinate<VALUETYPE>[A_csr.rows])));
		prevCoordinates = static_cast<Coordinate<VALUETYPE> *> (::operator new (sizeof(Coordinate<VALUETYPE>[A_csr.rows])));
		this->init = init;
	}
	void algorithms::randInit(){
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
	void algorithms::initDFS(){
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
		}
        }
	void algorithms::fileInitialization()
	{
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
				nCoordinates[index] = Coordinate <VALUETYPE>(x, y, i); 
				index++;
			}
		}
		fclose(infile);
	}
	Coordinate<VALUETYPE> algorithms::calcAttraction(INDEXTYPE i, INDEXTYPE j){
                Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
		for(INDEXTYPE n = graph.rowptr[i]; n < graph.rowptr[i+1]; n++){
			f = f + (this->nCoordinates[graph.colids[n]] - this->nCoordinates[i]) * (W * (this->nCoordinates[graph.colids[n]] - this->nCoordinates[i]).getMagnitude()) - calcRepulsion(i, graph.colids[n]);
		}
                return f;
        }

	Coordinate<VALUETYPE> algorithms::calcRepulsion(INDEXTYPE i, INDEXTYPE n){
		Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                if((this->nCoordinates[n] - this->nCoordinates[i]).getMagnitude2() > 0)
                	f = f - (this->nCoordinates[n] - this->nCoordinates[i]) * (1.0 /(this->nCoordinates[n] - this->nCoordinates[i]).getMagnitude2());
		return f;
	}
	
	VALUETYPE algorithms::updateStepLength(VALUETYPE STEP, VALUETYPE ENERGY, VALUETYPE ENERGY0){
		if(ENERGY < ENERGY0){
                        PROGRESS = PROGRESS + 1;
                        if(PROGRESS >= 5){
                                PROGRESS = 0;
                                STEP = STEP / t;
                        }
                }else{
                        PROGRESS = 0;
                        STEP = t * STEP;
                }
		return STEP;
	}
	
	//sequantial implementation of O(n^2) algorithm
	vector<VALUETYPE> algorithms::seqForceDirectedAlgorithm(INDEXTYPE ITERATIONS){
		INDEXTYPE LOOP = 0;
		VALUETYPE start, end, ENERGY, ENERGY0;
		VALUETYPE STEP = 1.0;
		vector<VALUETYPE> result;
		ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
		start = omp_get_wtime();
		if(init == 0){
                        initDFS();
                }else if(init == 2){
			fileInitialization();
		}
                else{
                        randInit();
                }
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
						f = f + (this->nCoordinates[j] - this->nCoordinates[i]) * (W * (this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude());
                                                if(k < graph.rowptr[i+1] - 1){
							k++;
						}
					}
                               		else{
						VALUETYPE dist = (this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude2();
						if(dist > 0.0)
                                                        f = f - (this->nCoordinates[j] - this->nCoordinates[i]) * (1.0 / dist);
					}
				}
				this->nCoordinates[i] = this->nCoordinates[i] + f.getUnitVector() * STEP;
                                ENERGY = ENERGY + f.getMagnitude2();
			}
			STEP = updateStepLength(STEP, ENERGY, ENERGY0);
			LOOP++;
		}
		end = omp_get_wtime();
		cout << "Sequential" << endl;
		cout << "Energy:" << ENERGY << endl;
		cout << "Sequential Wall time required:" << end - start << endl;
		result.push_back(ENERGY);
		result.push_back(end - start);
		writeToFile("SEQU" + to_string(ITERATIONS));
		return result;
	}
/*
 *
 *       Majedul: working on optimizing serial version first  
 */
	//sequantial implementation of O(n^2) algorithm
   vector<VALUETYPE> algorithms::opt_seqForceDirectedAlgorithm(INDEXTYPE ITERATIONS){
		INDEXTYPE LOOP = 0;
		VALUETYPE start, end, ENERGY, ENERGY0;
		VALUETYPE STEP = 1.0;
		vector<VALUETYPE> result;
		ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
		start = omp_get_wtime();
		if(init == 0){
                        initDFS();
                }else if(init == 2){
			fileInitialization();
		}
                else{
                        randInit();
                }
/*
 *             NOTE: how it works: 
 *                      rowptr (sorted, index --> starting and end of colids):
 *                        0 3 4 8 10 12      
 *                node(i):0 1 2 3  4  5       
 *
 *                      colids: 1 2 5  4  0 2 3 4   4 5   6 7  sorted in each group ???  
 *                              -----  -  -------   ---
 *                              0 1 2  3  4 5 6 7   8 9 
 */  
      while(LOOP < ITERATIONS)
      {
	 ENERGY0 = ENERGY;
         ENERGY = 0;

	 for(INDEXTYPE i = 0; i < graph.rows; i++)   /* test each node */
         {
            Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
	    INDEXTYPE k = graph.rowptr[i]; 
            
            Coordinate<VALUETYPE> iCord = this->nCoordinates[i]; 
            INDEXTYPE upper = graph.rowptr[i+1]-1 ;


            for(INDEXTYPE j = 0; j < graph.rows; j++) /* test i with all other nodes */
            {

               Coordinate<VALUETYPE> jCord = this->nCoordinates[j]; 
               Coordinate<VALUETYPE> dfCord = jCord - iCord; 
               VALUETYPE dist = dfCord.getMagnitude2();
               
               if (j != graph.colids[k])
               {
                  if (dist > 0)
                     f = f - dfCord * (1.0 / dist);
               }
               else if (j == graph.colids[k])
               {
	          f = f + dfCord * (W * dfCord.getMagnitude());
                  if(k < upper)
		     k++;
               }
/*
 *             make most frequent case as if case: non-connected case should be
 *             most frequent
 *
 *             NOTE:  we can apply speculative vectorization in most frequent
 *             case:
 *
 *             Pseudo code: 
 *          Vf = [Cordinate[0.0], .. ] ;
 *          for (j = 0; j < graph.rows; j+=4) // when vlen = 4
 *          {
 *             VjCord = nCord[j:j+3]; 
 *             VdfCord = Vjcord - [icord, icrod, icord, icord];
 *             Vdist = [VdfCord[0].getMag2, VdfCord[1].getMag2(), ...,...].
 *
 *             // speculative path 
 *             if ([j, j+1, j+2, j+3] != VEC_ALL (graph.colids[k, k, k, k] ) )
 *             {
 *                // note: using redundant comp... assuming most often dist > 0
 *                // if that not true.. speculation would be profitable here
 *                Vmask = Vdist > [0, 0, 0, 0] // 1 where the conditon is true 
 *                Vf1 = Vf - VdfCord * ([1,1,1,1]/Vdist);      
 *                Vf = select (Vf1, Vf, Vmask);
 *             }
 *             else // not vectorizable path due to update of K  
 *             {
 *                // assuming this path is less frequent .. when node is neighbor
 *                // if this path is frequent, we can futher apply speculation
 *                
 *                [f0, f1, f2, f3] = Vf; 
 *                if (j == graph.colids[k]) 
 *                {
	             f = f + dfCord * (W * dfCord.getMagnitude());
                     if (k < upper) k++; 
 *                }
 *                if (j+1 == graph.colids[k]) 
 *                {
	             f = f + dfCord * (W * dfCord.getMagnitude());
                     if (k < upper) k++; 
 *                }
 *                if (j+2 == graph.colids[k]) 
 *                {
	             f = f + dfCord * (W * dfCord.getMagnitude());
                     if (k < upper) k++; 
 *                }
 *                if (j+3 == graph.colids[k]) 
 *                {
	             f = f + dfCord * (W * dfCord.getMagnitude());
                     if (k < upper) k++; 
 *                }
 *                Vf = [f0, f1, f2, f3]; 
 *             }
 *          }
 */
	    }
	    this->nCoordinates[i] +=  (f.getUnitVector() * STEP);
            ENERGY = ENERGY + f.getMagnitude2();
	 }
	 STEP = updateStepLength(STEP, ENERGY, ENERGY0);
	 LOOP++;
      }
      end = omp_get_wtime();
		cout << "Sequential" << endl;
		cout << "Energy:" << ENERGY << endl;
		cout << "Sequential Wall time required:" << end - start << endl;
		result.push_back(ENERGY);
		result.push_back(end - start);
		writeToFile("SEQU" + to_string(ITERATIONS));
		return result;
	}
	
	vector<VALUETYPE> algorithms::seqAdjForceDirectedAlgorithm(INDEXTYPE ITERATIONS){
                INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
		vector<int> adjvect(graph.rows, 0);
                vector<VALUETYPE> result;
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                start = omp_get_wtime();
                if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }
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
                                                f = f + (this->nCoordinates[j] - this->nCoordinates[i]) * (W * (this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude()/K);
                                        }
                                        else{
                                                if((this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude2() > 0)
                        				f = f - (this->nCoordinates[j] - this->nCoordinates[i]) * (1.0 /(this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude2());
                                        }
                                }
                                this->nCoordinates[i] = this->nCoordinates[i] + f.getUnitVector() * STEP;
                                ENERGY = ENERGY + f.getMagnitude2();
                        }
                        STEP = updateStepLength(STEP, ENERGY, ENERGY0);
                        LOOP++;
                }
                end = omp_get_wtime();
		cout << "Adj Sequential" << endl;
                cout << "Energy Adj:" << ENERGY << endl;
                cout << "Sequential Adj Wall time required:" << end - start << endl;
                result.push_back(ENERGY);
                result.push_back(end - start);
                writeToFile("SEQUADJ" + to_string(ITERATIONS));
                return result;
        }
	
	vector<VALUETYPE> algorithms::naiveParallelForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS){
                INDEXTYPE LOOP = 0;
		VALUETYPE start, end, ENERGY, ENERGY0;
		VALUETYPE STEP = 1.0;
		vector<VALUETYPE> result;
		ENERGY = numeric_limits<VALUETYPE>::max();
		omp_set_num_threads(NUMOFTHREADS);
		start = omp_get_wtime();
		if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }
		while(LOOP < ITERATIONS){
			ENERGY0 = ENERGY;
			ENERGY = 0;
			Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
			INDEXTYPE j, k;
			for(INDEXTYPE i = 0; i < graph.rows; i++){
				f = Coordinate <VALUETYPE>(0.0, 0.0);
				#pragma omp parallel for reduction(plus:f)
				for(INDEXTYPE j = 0; j < graph.rows; j++){
					if(i == j){
						f += calcAttraction(i, j);
                                	}else{
						f += calcRepulsion(i, j);
                               		}
				}
				#
				this->nCoordinates[i] = this->nCoordinates[i] + f.getUnitVector() * STEP;
				ENERGY = ENERGY + f.getMagnitude2();
			}
			STEP = STEP * t;
			LOOP++;
		}
		end = omp_get_wtime();
		cout << "Naive parallel" << endl;
                cout << "Naive Energy:" << ENERGY << endl;
		result.push_back(ENERGY);
                cout << "Naive Parallel Wall time:" << end - start << endl;
		result.push_back(end - start);
                writeToFile("NAIVEPARA" + to_string(ITERATIONS));
		return result;
        }	
		
	vector<VALUETYPE> algorithms::miniBatchForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                srand(unsigned(time(0)));
		INDEXTYPE LOOP = 0;
                VALUETYPE start, end, ENERGY, ENERGY0, ATTRACTIVEENERGY;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                vector<INDEXTYPE> indices;
                for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
		if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }
		double FLOPS = 0.0;
                while(LOOP < ITERATIONS){
		//while((ENERGY0 - ENERGY)/ ENERGY0 < threshold && LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
			ATTRACTIVEENERGY = 0;
			//double s = omp_get_wtime();
			__gnu_parallel::random_shuffle(indices.begin(), indices.end());
			//double e = omp_get_wtime();
			//cout << "RS:" << e - s << endl;
			/*cout << LOOP << ":";
			for(int i=0; i < 10; i++){
				cout << indices[i] << " ";
			}
			cout << endl;
			*/
			FLOPS = 0.0;        
			for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static)	
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
					Coordinate<VALUETYPE> fattract = Coordinate <VALUETYPE>(0.0, 0.0);
                                        INDEXTYPE k = graph.rowptr[indices[i]];
                                        for(INDEXTYPE j = 0; j < graph.rows; j++){
						if(j == graph.colids[k]){
                                                	f += (nCoordinates[j] - nCoordinates[indices[i]]) * (W * (nCoordinates[j] - nCoordinates[indices[i]]).getMagnitude());
							//fattract += (nCoordinates[j] - nCoordinates[indices[i]]) * ((nCoordinates[j] - nCoordinates[indices[i]]).getMagnitude()/K);
                                                        if(k < graph.rowptr[indices[i]+1]-1){
                                                                k++;
                                                        }
						}else{
                                                        if((this->nCoordinates[j] - this->nCoordinates[indices[i]]).getMagnitude2() > 0)
                        				{
								f = f - (this->nCoordinates[j] - this->nCoordinates[indices[i]]) * (1.0 /(this->nCoordinates[j] - this->nCoordinates[indices[i]]).getMagnitude2());
								//fattract += (this->nCoordinates[j] - this->nCoordinates[indices[i]]) * (C * K * K /(this->nCoordinates[j] - this->nCoordinates[indices[i]]).getMagnitude2());
							}
                                                }
                                        }
                                        prevCoordinates[indices[i]] = f;
                                        //ENERGY += f.getMagnitude2();
					//ATTRACTIVEENERGY += fattract.getMagnitude2();
				}
				//#pragma omp for simd schedule(static) 
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[indices[i]] = nCoordinates[indices[i]] + prevCoordinates[indices[i]].getUnitVector() * STEP;
                                	ENERGY += prevCoordinates[indices[i]].getMagnitude2();
				}
                        }
			STEP = STEP * 0.999;
                        LOOP++;
                }
		end = omp_get_wtime();
                cout << "Final Minibatch Size:" << BATCHSIZE << endl;
                cout << "FinalMinbatch Energy:" << ENERGY << endl;
                result.push_back(ENERGY);
                cout << "Final Minibatch Parallel Wall time required:" << end - start << endl;
                result.push_back(end - start);
                writeToFile("MINB"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(ITERATIONS));
                return result;
	}

	vector<VALUETYPE> algorithms::cacheBlockingminiBatchForceDirectedAlgorithmSD(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, int flag = 0){
		INDEXTYPE LOOP = 0;
                INDEXTYPE blocky = 512, blockx = 2;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                vector<INDEXTYPE> indices;
                vector<int> kindex(graph.rows, 0);
                for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                if(flag == 0){
                if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }}else{
                        STEP = pow(0.999, 4 * ITERATIONS);
                }
		while(LOOP < ITERATIONS){
			ENERGY0 = ENERGY;
                        ENERGY = 0;
                        #pragma omp parallel for simd
                        for(INDEXTYPE k = 0; k < graph.rows; k++){
                                prevCoordinates[k] = Coordinate <VALUETYPE>(0.0, 0.0);
                        }
                        for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += 1){
                                        if(i >= graph.rows)continue;
					//#pragma omp simd
					for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j += 1){
						int v = graph.colids[j];
						prevCoordinates[i] += (this->nCoordinates[v] - this->nCoordinates[i]) * (W * (this->nCoordinates[v] - this->nCoordinates[i]).getMagnitude()) + (this->nCoordinates[v] - this->nCoordinates[i]) * (1.0 / ((this->nCoordinates[v] - this->nCoordinates[i]).getMagnitude2()));
						/*if(!(prevCoordinates[i].x)){
                                                printf("i = %d, j = %d <-->\n", i, v);

                                                exit(1);
                                        	}*/
					}
					Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
					//#pragma omp simd
                                        for(INDEXTYPE j = 0; j < i; j += 1){
						f += (this->nCoordinates[j] - this->nCoordinates[i]) * (1.0 / ((this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude2()));
                                        	/*if(!(f.x)){
                                                printf("i = %d, j = %d\n", i, j);
                                                exit(1);
                                        	}*/
					}
					//#pragma omp simd
					for(INDEXTYPE j = i+1; j < graph.rows; j += 1){
						f += (this->nCoordinates[j] - this->nCoordinates[i]) * (1.0 / ((this->nCoordinates[j] - this->nCoordinates[i]).getMagnitude2()));
						/*if(!f.x){
                                                printf("i = %d, j = %d\n", i, j);
                                                exit(1);
                                                }*/
					}
					prevCoordinates[i] = prevCoordinates[i] - f;
					/*if(!prevCoordinates[i].x){
						printf("i = %d\n", i);
						exit(1);
					}*/
                                }
				/*for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
					cout << "Node:" << i << ", X:" << prevCoordinates[i].getX() << ", Y:" << prevCoordinates[i].getY()<< endl;
				}
				cout << endl;
                                */
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) break;
                                        nCoordinates[i] = nCoordinates[i] + prevCoordinates[i].getUnitVector() * STEP;
                                        ENERGY += prevCoordinates[i].getMagnitude2();
                                }
				//print();
                        }
			STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
                if(flag == 0){
                cout << "Cache BlockingSD Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Cache BlockingSD Minbatch Energy:" << ENERGY << endl;
                cout << "Cache BlockingSD Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFile("CACHESDMINB"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
                }
                result.push_back(ENERGY);
                result.push_back(end - start);
		return result;
	}
	vector<VALUETYPE> algorithms::cacheBlockingminiBatchForceDirectedAlgorithmVD(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, int flag = 0){
		INDEXTYPE LOOP = 0;
                INDEXTYPE blocky = 512, blockx = 2;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                vector<INDEXTYPE> indices;
                vector<int> kindex(graph.rows, 0);
		ENERGY0 = numeric_limits<VALUETYPE>::max();
                ENERGY = 0;
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                if(flag == 0){
                if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{   
                        randInit();
                }}else{
                        STEP = pow(0.999, 4 * ITERATIONS);
                }
		//printf("OK1\n");
                while(LOOP < ITERATIONS){	
			ENERGY0 = ENERGY;
                        ENERGY = 0;
                        #pragma omp parallel for simd proc_bind(close)
                        for(INDEXTYPE k = 0; k < graph.rows; k++){
                                kindex[k] = graph.rowptr[k];
                                prevCoordinates[k] = Coordinate <VALUETYPE>(0.0, 0.0);
                        }
                        for(INDEXTYPE b = 0; b < (int)(graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += blockx){
                                        for(INDEXTYPE j = 0; j < graph.rows; j += blocky){
                                                for(INDEXTYPE bi = 0; bi < blockx && i + bi < (b + 1) * BATCHSIZE; bi++){
                                                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
							//printf("OK2\n");
                                                        for(INDEXTYPE bj = 0; bj < blocky && j + bj < graph.rows; bj++){
                                                                if(j + bj == graph.colids[kindex[bi+i]]){
                                                                        f += (nCoordinates[j+bj] - nCoordinates[i+bi]) * (W * (nCoordinates[j+bj] - nCoordinates[i+bi]).getMagnitude());
                                                                        if(kindex[bi+i] < graph.rowptr[i+bi+1] - 1){
                                                                                kindex[bi+i]++;
                                                                        }
                                                                }else{
                                                                        VALUETYPE dist = (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]).getMagnitude2();
                                                                        if(dist > 0)
                                                                        {
                                                                                f = f - (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]) * (1.0 / (dist));
                                                                        }
                                                                }
                                                        }
                                                        prevCoordinates[i+bi] += f;
                                                }
                                        }
                                }
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        nCoordinates[i] = nCoordinates[i] + prevCoordinates[i].getUnitVector() * STEP;
                                        ENERGY += prevCoordinates[i].getMagnitude2();
                                }
                        }
			//printf("OK2\n");	
                        #pragma omp parallel for schedule(static)
                        for(INDEXTYPE i = (int)(graph.rows / BATCHSIZE) * BATCHSIZE; i < graph.rows; i += blockx){
                        	for(INDEXTYPE j = 0; j < graph.rows; j += blocky){
                                	for(INDEXTYPE bi = 0; bi < blockx && i + bi < graph.rows; bi++){                                         
                                        	Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                                                for(INDEXTYPE bj = 0; bj < blocky && j + bj < graph.rows; bj++){
                                                	if(j + bj == graph.colids[kindex[bi+i]]){
                                                        	f += (nCoordinates[j+bj] - nCoordinates[i+bi]) * (W * (nCoordinates[j+bj] - nCoordinates[i+bi]).getMagnitude());
                                                                if(kindex[bi+i] < graph.rowptr[i+bi+1] - 1){
                                                                	kindex[bi+i]++;
                                                                }
                                             		}else{
                                                        	VALUETYPE dist = (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]).getMagnitude2();
                                                                if(dist > 0)
                                                                {
                                                                	f = f - (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]) * (1.0 / (dist));
                                                                }
                                                        }
                                                  }
                                                  prevCoordinates[i+bi] += f;
                                        }
                                }
                        }
                        for(INDEXTYPE i = (int)(graph.rows / BATCHSIZE) * BATCHSIZE; i < graph.rows; i++){
                        	nCoordinates[i] = nCoordinates[i] + prevCoordinates[i].getUnitVector() * STEP;
                                ENERGY += prevCoordinates[i].getMagnitude2();
                       	}
                        STEP = STEP * 0.999;
                        LOOP++;
                }
                end = omp_get_wtime();
		if(flag == 0){
                cout << "Vectorized Cache Blocking Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Vectorized Cache Blocking Minbatch Energy:" << ENERGY << endl;
                cout << "Vectorized Cache Blocking Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFile("VCACHEMINB"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
                }
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
	}
	vector<VALUETYPE> algorithms::cacheBlockingminiBatchForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, int flag = 0){
                INDEXTYPE LOOP = 0;
		INDEXTYPE blocky = 512, blockx = 2;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                vector<INDEXTYPE> indices;
		vector<int> kindex(graph.rows, 0);
                //for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
                ENERGY0 = numeric_limits<VALUETYPE>::max();
                ENERGY = 0;
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
		if(flag == 0){
		if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }}else{
			STEP = pow(0.999, 4 * ITERATIONS);
		}
		while(LOOP < ITERATIONS){
		//while((fabs(ENERGY0 - ENERGY)/ENERGY0 > threshold)){
                        //printf("%d %lf\n", LOOP, ENERGY);
			ENERGY0 = ENERGY;
                        ENERGY = 0;
			#pragma omp parallel for simd proc_bind(close)
                        for(INDEXTYPE k = 0; k < graph.rows; k++){
                        	kindex[k] = graph.rowptr[k];
                        	prevCoordinates[k] = Coordinate <VALUETYPE>(0.0, 0.0);
			}
			for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
				#pragma omp parallel for schedule(static)
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += blockx){
                                	if(i >= graph.rows)continue;
                                	for(INDEXTYPE j = 0; j < graph.rows; j += blocky){
						for(INDEXTYPE bi = 0; bi < blockx && i + bi < (b + 1) * BATCHSIZE; bi++){
							if(i+bi >= graph.rows) break;
							Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                                        		for(INDEXTYPE bj = 0; bj < blocky && j + bj < graph.rows; bj++){
								if(j + bj == graph.colids[kindex[bi+i]]){
									f += (nCoordinates[j+bj] - nCoordinates[i+bi]) * (W * (nCoordinates[j+bj] - nCoordinates[i+bi]).getMagnitude());
									if(kindex[bi+i] < graph.rowptr[i+bi+1] - 1){
										kindex[bi+i]++;
									}
								}else{
									VALUETYPE dist = (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]).getMagnitude2();
									if(dist > 0)
                                                        		{
                                                                		f = f - (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]) * (1.0 / (dist));
                                                        		}
								}
							}
							prevCoordinates[i+bi] += f;
						}
					}
				}
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[i] = nCoordinates[i] + prevCoordinates[i].getUnitVector() * STEP;
                                	ENERGY += prevCoordinates[i].getMagnitude2();
				}
			}
                        STEP = STEP * 0.999;
			LOOP++;
                }
                end = omp_get_wtime();
		if(flag == 0){
                cout << "Cache Blocking Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Cache Blocking Minbatch Energy:" << ENERGY << endl;
                cout << "Cache Blocking Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFile("CACHEMINB"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
                }
		result.push_back(ENERGY);
		result.push_back(end - start);
		return result;
        }
	vector<VALUETYPE> algorithms::cacheBlockingminiBatchForceDirectedAlgorithmConverged(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, int flag = 0){
                INDEXTYPE LOOP = 0;
                INDEXTYPE blocky = 512, blockx = 2;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                vector<INDEXTYPE> indices;
                vector<int> kindex(graph.rows, 0);
		ENERGY0 = numeric_limits<VALUETYPE>::max();
                ENERGY = 0;
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                if(flag == 0){
                if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }}else{
                        STEP = pow(0.999, 4 * ITERATIONS);
                }
                while((fabs(ENERGY0 - ENERGY)/ENERGY0 > threshold)){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        #pragma omp parallel for simd proc_bind(close)
                        for(INDEXTYPE k = 0; k < graph.rows; k++){
                                kindex[k] = graph.rowptr[k];
                                prevCoordinates[k] = Coordinate <VALUETYPE>(0.0, 0.0);
                        }
                        for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += blockx){
                                        if(i >= graph.rows)continue;
                                        for(INDEXTYPE j = 0; j < graph.rows; j += blocky){
                                                for(INDEXTYPE bi = 0; bi < blockx && i + bi < (b + 1) * BATCHSIZE; bi++){
                                                        if(i+bi >= graph.rows) break;
                                                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                                                        for(INDEXTYPE bj = 0; bj < blocky && j + bj < graph.rows; bj++){
                                                                if(j + bj == graph.colids[kindex[bi+i]]){
                                                                        f += (nCoordinates[j+bj] - nCoordinates[i+bi]) * (W * (nCoordinates[j+bj] - nCoordinates[i+bi]).getMagnitude());
                                                                        if(kindex[bi+i] < graph.rowptr[i+bi+1] - 1){
                                                                                kindex[bi+i]++;
                                                                        }
                                                                }else{
                                                                        VALUETYPE dist = (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]).getMagnitude2();
                                                                        if(dist > 0)
                                                                        {
                                                                                f = f - (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]) * (1.0 / (dist));
                                                                        }
                                                                }
                                                        }
                                                        prevCoordinates[i+bi] += f;
                                                }
                                        }
                                }
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[i] = nCoordinates[i] + prevCoordinates[i].getUnitVector() * STEP;
                                        ENERGY += prevCoordinates[i].getMagnitude2();
                                }
                        }
                        STEP = STEP * 0.999;
                        LOOP++;
                }
		end = omp_get_wtime();
                if(flag == 0){
                cout << "Cache Blocking (converged) Minibatch Size:" << BATCHSIZE  << endl;
                cout << "Cache Blocking (converged)  Minbatch Energy:" << ENERGY << endl;
                cout << "Cache Blocking (converged) Minibatch Parallel Wall time required:" << end - start << endl;
                writeToFile("CACHEMINB"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(LOOP));
                }
                result.push_back(ENERGY);
                result.push_back(end - start);
                return result;
        }


	vector<VALUETYPE> algorithms::LinLogcacheBlockingminiBatchForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0;
                INDEXTYPE blocky = 512, blockx = 2;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                vector<INDEXTYPE> indices;
                vector<int> kindex(graph.rows, 0);
                for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }
		while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        #pragma omp parallel for simd schedule(static)
                        for(INDEXTYPE k = 0; k < graph.rows; k++){
                                kindex[k] = graph.rowptr[k];
				prevCoordinates[k] = Coordinate <VALUETYPE>(0.0, 0.0);
                        }
                        for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += blockx){
                                        if(i >= graph.rows)continue;
                                        for(INDEXTYPE j = 0; j < graph.rows; j += blocky){
                                                for(INDEXTYPE bi = 0; bi < blockx && i + bi < (b + 1) * BATCHSIZE; bi++){
                                                        if(i+bi >= graph.rows) break;
                                                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                                                        for(INDEXTYPE bj = 0; bj < blocky && j + bj < graph.rows; bj++){
                                                                if(j + bj == graph.colids[kindex[bi+i]]){
                                                                        f += (nCoordinates[j+bj] - nCoordinates[i+bi]) * log2(1.0 + W * (nCoordinates[j+bj] - nCoordinates[i+bi]).getMagnitude());
                                                                        if(kindex[bi+i] < graph.rowptr[i+bi+1] - 1){
                                                                                kindex[bi+i]++;
                                                                        }
                                                                }else{
									VALUETYPE dist = (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]).getMagnitude2();
                                                                        if(dist > 0)
                                                                        {
                                                                                f = f - (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]) * (1.0 / dist);
                                                                        }
                                                                }
                                                        }
                                                        prevCoordinates[i+bi] += f;
                                                }
                                        }
                                }
				for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[i] = nCoordinates[i] + prevCoordinates[i].getUnitVector() * STEP;
                                        ENERGY += prevCoordinates[i].getMagnitude2();
                                }
                        }
                        STEP = STEP * t;
                        LOOP++;
                }
                end = omp_get_wtime();
                cout << "LinLog Batch Size:" << BATCHSIZE << endl;
                cout << "LinLog Minbatch Energy:" << ENERGY << endl;
                result.push_back(ENERGY);
                cout << "LinLog Minibatch Parallel Wall time required:" << end - start << endl;
                result.push_back(end - start);
                writeToFile("LLCACHEMINB"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(ITERATIONS));
                return result;
        }

	vector<VALUETYPE> algorithms::FAcacheBlockingminiBatchForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE LOOP = 0;
                INDEXTYPE blocky = 512, blockx = 2;
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
                vector<INDEXTYPE> indices;
                vector<int> kindex(graph.rows, 0);
                for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }
                while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        #pragma omp parallel for simd schedule(static)
                        for(INDEXTYPE k = 0; k < graph.rows; k++){
                                kindex[k] = graph.rowptr[k];
                                prevCoordinates[k] = Coordinate <VALUETYPE>(0.0, 0.0);
                        }
                        for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static)
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i += blockx){
                                        if(i >= graph.rows)continue;
                                        for(INDEXTYPE j = 0; j < graph.rows; j += blocky){
                                                for(INDEXTYPE bi = 0; bi < blockx && i + bi < (b + 1) * BATCHSIZE; bi++){
                                                        if(i+bi >= graph.rows) break;
                                                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
                                                        for(INDEXTYPE bj = 0; bj < blocky && j + bj < graph.rows; bj++){
                                                                if(j + bj == graph.colids[kindex[bi+i]]){
                                                                        f += (nCoordinates[j+bj] - nCoordinates[i+bi]) * W;
                                                                        if(kindex[bi+i] < graph.rowptr[i+bi+1] - 1){
                                                                                kindex[bi+i]++;
                                                                        }
                                                                }else{
                                                                        VALUETYPE dist = (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]).getMagnitude2();
									if(dist > 0)
                                                                        {
                                                                                f = f - (this->nCoordinates[j+bj] - this->nCoordinates[i+bi]) * (1.0 / dist);
                                                                        }
                                                                }
                                                        }
                                                        prevCoordinates[i+bi] += f;
                                                }
                                        }
                                }
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[i] = nCoordinates[i] + prevCoordinates[i].getUnitVector() * STEP;
                                        ENERGY += prevCoordinates[i].getMagnitude2();
                                }
                        }
                        STEP = STEP * t;
                        LOOP++;
                }
                end = omp_get_wtime();
                cout << "FA Batch Size:" << BATCHSIZE << endl;
                cout << "FA Minbatch Energy:" << ENERGY << endl;
                result.push_back(ENERGY);
                cout << "FA Minibatch Parallel Wall time required:" << end - start << endl;
                result.push_back(end - start);
                writeToFile("FACACHEMINB"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(ITERATIONS));
                return result;
        }
		
	vector<VALUETYPE> algorithms::BarnesHutApproximation(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE, VALUETYPE TH, int flag = 0){
                INDEXTYPE LOOP = 0;
		VALUETYPE start, end, ENERGY, ENERGY0;
		vector<VALUETYPE> result;
		VALUETYPE STEP = 1.0;
		ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
		Coordinate<VALUETYPE> *tempCoordinates = static_cast<Coordinate<VALUETYPE> *> (::operator new (sizeof(Coordinate<VALUETYPE>[graph.rows])));
                omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
		if(flag == 0){
		if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }}
		while(LOOP < ITERATIONS){
			ENERGY0 = ENERGY;
                        ENERGY = 0;
			#pragma omp parallel for simd schedule(static)
                        for(INDEXTYPE k = 0; k < graph.rows; k++){
				prevCoordinates[k] = Coordinate <VALUETYPE>(0.0, 0.0);
				tempCoordinates[k] = nCoordinates[k];
                        }
			//VALUETYPE s = omp_get_wtime();
			BarnesHut bh(tempCoordinates, graph.rows, TH);
			//VALUETYPE e = omp_get_wtime();
			//printf("BH Time: %lf\n", e - s);
			for(INDEXTYPE b = 0; b < (int)ceil( 1.0 * graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static) 
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
					#pragma omp simd
                                        for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j++){
						f = f + (nCoordinates[graph.colids[j]] - nCoordinates[i]) * (W * (nCoordinates[graph.colids[j]] - nCoordinates[i]).getMagnitude());
					}
					f = f - bh.calcRepForce(nCoordinates[i]);
					prevCoordinates[i] = f;
				}
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[i] = nCoordinates[i] + prevCoordinates[i].getUnitVector() * STEP;
                                        ENERGY += prevCoordinates[i].getMagnitude2();
                                }
				
			}
			//STEP = updateStepLength(STEP, ENERGY, ENERGY0);
                        STEP = STEP * 0.999;
			LOOP++;
		}
		end = omp_get_wtime();
		if(flag == 0){
                cout << "Barnes-Hut Minibatch Size:" << BATCHSIZE << endl;
                cout << "Barnes-Hut Minbatch Energy:" << ENERGY << endl;
                result.push_back(ENERGY);
                cout << "Barnes-Hut Minibatch Parallel Wall time required:" << end - start << endl;
                result.push_back(end - start);
                writeToFile("BHMINB"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(ITERATIONS));
		}
                return result;
	}

	vector<VALUETYPE> algorithms::approxForceDirectedAlgorithm(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
		INDEXTYPE LOOP = 0, approxITER = (int)(ITERATIONS * 0.8);
                VALUETYPE start, end, ENERGY, ENERGY0;
                VALUETYPE STEP = 1.0;
                vector<VALUETYPE> result;
		vector<INDEXTYPE> indices;
                for(INDEXTYPE i = 0; i < graph.rows; i++) indices.push_back(i);
                ENERGY0 = ENERGY = numeric_limits<VALUETYPE>::max();
		omp_set_num_threads(NUMOFTHREADS);
                start = omp_get_wtime();
                if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }
		while(LOOP < ITERATIONS){
                        ENERGY0 = ENERGY;
                        ENERGY = 0;
                        INDEXTYPE j;
                        INDEXTYPE k;
                        for(INDEXTYPE b = 0; b < (int)ceil(1.0 * graph.rows / BATCHSIZE); b += 1){
                                #pragma omp parallel for schedule(static)   
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        Coordinate<VALUETYPE> f = Coordinate <VALUETYPE>(0.0, 0.0);
					stack <int> STACKnode;
					if(LOOP > approxITER){
						INDEXTYPE k = graph.rowptr[indices[i]];
                                        	for(INDEXTYPE j = 0; j < graph.rows; j++){
                                                	if(j == graph.colids[k] && k < graph.nnz){
                                                        	f += (nCoordinates[j] - nCoordinates[indices[i]]) * (W * (nCoordinates[j] - nCoordinates[indices[i]]).getMagnitude());
                                                        	if(k < graph.rowptr[indices[i]+1]-1){
                                                                	k++;
                                                        	}
                                                	}else{
								VALUETYPE dist = (this->nCoordinates[j] - this->nCoordinates[indices[i]]).getMagnitude();
								if(dist > 0)
                                                        	{
                                                                	f = f - (this->nCoordinates[j] - this->nCoordinates[indices[i]]) * (1.0 / dist);
                                                        	}
                                                	}
                                        	}
					}
					else{
						unordered_map<int, int> neighbors;
                                        	neighbors.insert(pair<int, int>(indices[i], indices[i]));
						if(i < graph.rows - 1){
                        				for(INDEXTYPE j = graph.rowptr[i]; j < graph.rowptr[i+1]; j++){
								f += (nCoordinates[graph.colids[j]] - nCoordinates[indices[i]]) * (W * (nCoordinates[graph.colids[j]] - nCoordinates[indices[i]]).getMagnitude());
								STACKnode.push(graph.colids[j]);
								neighbors.insert(pair<int, int>(graph.colids[j], indices[i]));
                        				}
                				}else{
                        				for(INDEXTYPE j = graph.rowptr[i]; j < graph.nnz; j++){
								f += (nCoordinates[graph.colids[j]] - nCoordinates[indices[i]]) * (W * (nCoordinates[graph.colids[j]] - nCoordinates[indices[i]]).getMagnitude());
                                                        	STACKnode.push(graph.colids[j]);
                                                       		neighbors.insert(pair<int, int>(graph.colids[j], indices[i]));
                        				}
                				}
						int countNodes = 200;
						while(!STACKnode.empty()){
							int currentn = STACKnode.top();
							STACKnode.pop();
							if(currentn < graph.rows - 1){
                                                 	       for(INDEXTYPE n = graph.rowptr[currentn]; n < graph.rowptr[currentn+1]; n++){
                                                        	        if(neighbors.count(graph.colids[n]) < 1){
                                                                	        f += calcRepulsion(indices[i], graph.colids[n]);
                                                                        	STACKnode.push(graph.colids[n]);
                                                                        	countNodes--;
                                                                	}
                                                        	}
                                                	}else{
                                                        	for(INDEXTYPE n = graph.rowptr[currentn]; n < graph.nnz; n++){
                                                                	if(neighbors.count(graph.colids[n]) < 1){
                                                                        	f += calcRepulsion(indices[i], graph.colids[n]);
                                                                        	STACKnode.push(graph.colids[n]);
                                                                        	countNodes--;
                                                                	}
                                                        	}
                                                	}
							if(countNodes <= 0)break;
						}
                                	}
					prevCoordinates[indices[i]] = f;
				}
                                for(INDEXTYPE i = b * BATCHSIZE; i < (b + 1) * BATCHSIZE; i++){
                                        if(i >= graph.rows) continue;
                                        nCoordinates[indices[i]] = nCoordinates[indices[i]] + prevCoordinates[indices[i]].getUnitVector() * STEP;
                                	ENERGY += prevCoordinates[indices[i]].getMagnitude2();
				}
                        }
			STEP = STEP * t;
                        LOOP++;
                }
                end = omp_get_wtime();
		cout << "Greedy Approximation" << endl;
                cout << "Greedy Approximation Energy:" << ENERGY << endl;
                cout << "Greedy Approximation Wall time required:" << end - start << endl;
                result.push_back(ENERGY);
                result.push_back(end - start);
		writeToFile("GAPPROX"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(ITERATIONS));
                return result;
        }	
	
	vector<VALUETYPE> algorithms::approxCacheBlockBH(INDEXTYPE ITERATIONS, INDEXTYPE NUMOFTHREADS, INDEXTYPE BATCHSIZE){
                INDEXTYPE approxITER = (int)(ITERATIONS * 0.8);
		VALUETYPE start, end;
		vector<VALUETYPE> result;
		start = omp_get_wtime();
		if(init == 0){
                        initDFS();
                }else if(init == 2){
                        fileInitialization();
                }
                else{
                        randInit();
                }
		BarnesHutApproximation(approxITER, NUMOFTHREADS, BATCHSIZE, 1.2, 1);
		result = cacheBlockingminiBatchForceDirectedAlgorithm(ITERATIONS-approxITER, NUMOFTHREADS, BATCHSIZE, 1);	
		end = omp_get_wtime();
		result[1] = end - start;
		cout << "80%% - 20%% BH - CB" << endl;
		cout << "BH-CACHE Approximation Energy:" << result[0] << endl;
                cout << "BH-CACHE Approximation Wall time required:" << end - start << endl;
		writeToFile("BCAPPROX"+ to_string(BATCHSIZE)+"PARAOUT" + to_string(ITERATIONS));
		return result;
		
	}

	void algorithms::print(){
		for(INDEXTYPE i = 0; i < graph.rows; i++){
                	cout << "Node:" << i << ", X:" << nCoordinates[i].getX() << ", Y:" << nCoordinates[i].getY()<< endl;
        	}
		cout << endl;
	}
	void algorithms::writeRepulsiveForce(vector<Coordinate<VALUETYPE> > &repulse, string f){
		ofstream output;
		output.open(f);
		for(INDEXTYPE i = 0; i < graph.rows; i++){
			output << repulse[i].getMagnitude2() << "\t" << repulse[i].getX() << "\t" << repulse[i].getY() << endl;
		}
		output.close();
	}
	void algorithms::writeToFileBH(Coordinate<VALUETYPE> *tCoordinates, string f){
		stringstream  data(filename);
                string lasttok;
                while(getline(data,lasttok,'/'));
                filename = outputdir + lasttok + f + ".txt";
                ofstream output;
                output.open(filename);
                cout << "Creating output file in following directory:" << filename << endl;
                for(INDEXTYPE i = 0; i < graph.rows; i++){
                        output << tCoordinates[i].getX() <<"\t"<< tCoordinates[i].getY() << "\t" << i+1 << endl;
                }
                output.close();
	}
	void algorithms::writeToFile(string f){
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
