BatchLayout = . 
SAMPLE = ./sample
UNITTEST = ./Test
BIN = ./bin
INCDIR = -I$(BatchLayout) -I$(SAMPLE) -I$(UNITTEST)

COMPILER = g++
FLAGS = -g -fopenmp -O3 -std=c++11 -fpermissive -DCPP

$(warning SAMPLE is $(SAMPLE))

algorithms.o:	$(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h IO.h CSR.h CSC.h 
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/algorithms.o $(SAMPLE)/algorithms.cpp

batchlayout.o:	$(UNITTEST)/BatchLayout.cpp $(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/batchlayout.o $(UNITTEST)/BatchLayout.cpp

all:	algorithms.o batchlayout.o
	$(COMPILER) $(INCDIR) $(FLAGS) -g -o $(BIN)/BatchLayout $(BIN)/algorithms.o $(BIN)/batchlayout.o
	
clean:
	rm -rf ./bin/*
