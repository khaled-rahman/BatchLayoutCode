BatchLayout = . 
SAMPLE = ./sample
UNITTEST = ./Test
BIN = ./bin
Kernel = ./kernels
INCDIR = -I$(BatchLayout) -I$(SAMPLE) -I$(UNITTEST) -I$(Kernel)

COMPILER = g++

FLAGS = -g -fomit-frame-pointer -fopenmp -ffast-math -mavx512f -mavx512dq -O3 -std=c++11 -DCPP
#FLAGS = -g -fopenmp -O3 -std=c++11 -DCPP

all: batchlayout

algorithms.o:	$(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h IO.h CSR.h CSC.h 
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/algorithms.o $(SAMPLE)/algorithms.cpp

nblas.o:	$(Kernel)/nblas.cpp $(Kernel)/nblas.h IO.h CSR.h CSC.h
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/nblas.o $(Kernel)/nblas.cpp

newalgo.o:	$(SAMPLE)/newalgo.cpp $(SAMPLE)/newalgo.h IO.h CSR.h CSC.h $(Kernel)/nblas.cpp
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/newalgo.o $(SAMPLE)/newalgo.cpp

batchlayout.o:	$(UNITTEST)/BatchLayout.cpp $(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h $(Kernel)/nblas.cpp $(Kernel)/nblas.h
		$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/batchlayout.o $(UNITTEST)/BatchLayout.cpp

batchlayout:	algorithms.o batchlayout.o
		$(COMPILER) $(INCDIR) $(FLAGS) -o $(BIN)/BatchLayout $(BIN)/algorithms.o $(BIN)/batchlayout.o

utest.o:	$(UNITTEST)/utest.cpp $(SAMPLE)/algorithms.cpp $(SAMPLE)/algorithms.h $(SAMPLE)/newalgo.h $(SAMPLE)/newalgo.cpp $(Kernel)/nblas.cpp
	$(COMPILER) $(INCDIR) $(FLAGS) -c -o $(BIN)/utest.o $(UNITTEST)/utest.cpp

utest:	algorithms.o newalgo.o utest.o nblas.o
	$(COMPILER) $(INCDIR) $(FLAGS) -o $(BIN)/utest $(BIN)/algorithms.o $(BIN)/utest.o $(BIN)/newalgo.o $(BIN)/nblas.o


clean:
	rm -rf ./bin/*
