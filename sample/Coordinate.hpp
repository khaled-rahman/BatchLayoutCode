#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cmath>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <random>

//#include "overridenew.h"
#include "../utility.h"
#include "../CSC.h"
#include "../CSR.h"
#include "../IO.h"
//
using namespace std;
extern "C" {
#include "../GTgraph/R-MAT/defs.h"
#include "../GTgraph/R-MAT/init.h"
#include "../GTgraph/R-MAT/graph.h"
}


#ifdef CPP
#define MALLOC "new"
#endif

#define ITERS 10

template <class VALUETYPE>
class Coordinate{
	//VALUETYPE x, y;
	public:
		VALUETYPE x, y;
		Coordinate(){
			this->x = 0;
			this->y = 0;
		}
		Coordinate(VALUETYPE x, VALUETYPE y){
			this->x = x;
			this->y = y;
		}
		VALUETYPE getX(){
			return this->x;
		}
		VALUETYPE getY(){
			return this->y;
		}
		VALUETYPE getMagnitude(){
			return (VALUETYPE)sqrt(this->x * this->x + this->y * this->y);
		}
		VALUETYPE getMagnitude2(){
			return (VALUETYPE)(this->x * this->x + this->y * this->y);
		}
		VALUETYPE getDistance(Coordinate A){
			return (VALUETYPE)sqrt((this->x - A.x)*(this->x - A.x) + (this->y - A.y)*(this->y - A.y));
		}
		Coordinate getUnitVector(){
			return Coordinate(this->x / getMagnitude(), this->y / getMagnitude());
		}
		Coordinate operator*(VALUETYPE v){
			return Coordinate(this->x * v, this->y * v);
		}
		Coordinate operator/(VALUETYPE v){
			return Coordinate(this->x / v, this->y / v);
		}
		Coordinate operator+(Coordinate A){
			return Coordinate(this->x + A.x, this->y + A.y);
		}
		Coordinate operator-(Coordinate A){
			return Coordinate(this->x - A.x, this->y - A.y);
		}
		void operator+=(Coordinate A){
			this->x += A.x;
			this->y += A.y;
		}
		//#pragma omp declare reduction(plus:Coordinate<VALUETYPE>:omp_out += omp_in) initializer(omp_priv(omp_orig))
};

template<class VALUETYPE>
VALUETYPE get_random(VALUETYPE lowerbound, VALUETYPE upperbound){
	return lowerbound + (upperbound-lowerbound) * static_cast <VALUETYPE> (random()) / static_cast <VALUETYPE> (RAND_MAX);
}

template<class VALUETYPE>
VALUETYPE get_fixed_random(VALUETYPE lowerbound, VALUETYPE upperbound){
	return lowerbound + (VALUETYPE)fmod(rand(), (upperbound - lowerbound + 1));
}
