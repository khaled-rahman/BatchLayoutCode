BatchLayout = . 
SAMPLE = ./sample
UNITTEST = ./Test
BIN = ./bin
INCDIR = -I$(BatchLayout) -I$(SAMPLE) -I$(UNITTEST)

COMPILER = g++
FLAGS = -g -fopenmp -ffast-math -mavx512f -mavx512dq -O3 -std=c++11 -DCPP
#FLAGS = -g -fopenmp -O3 -std=c++11 -DCPP
all: batchlayout

algorithms.o:	$(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h IO.h CSR.h CSC.h 
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/algorithms.o $(SAMPLE)/algorithms.cpp

batchlayout.o:	$(UNITTEST)/BatchLayout.cpp $(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/batchlayout.o $(UNITTEST)/BatchLayout.cpp

batchlayout:	algorithms.o batchlayout.o
		$(COMPILER) $(INCDIR) $(FLAGS) -g -o $(BIN)/BatchLayout $(BIN)/algorithms.o $(BIN)/batchlayout.o
	
clean:
	rm -rf ./bin/*
