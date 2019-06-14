RMATPATH = GTgraph/R-MAT
SPRNPATH = GTgraph/sprng2.0-lite

include GTgraph/Makefile.var
INCLUDE += -I$(SPRNPATH)/include
COMPILER = g++

FLAGS = -g -fopenmp -O3 -std=c++11


#-lnuma -lmemkind -lpthread
#-qopt-report=5 -xMIC-AVX512 

sprng:	
	(cd $(SPRNPATH); $(MAKE); cd ../..)

rmat:	sprng
	(cd $(RMATPATH); $(MAKE); cd ../..)

TOCOMPILE = $(RMATPATH)/graph.o $(RMATPATH)/utils.o $(RMATPATH)/init.o $(RMATPATH)/globals.o 

# flags defined in GTgraph/Makefile.var

SAMPLE = ./sample
UNITTEST = ./unitTest
BIN = ./bin
SRC_SAMPLE = $(wildcard $(SAMPLE)/*.cpp)
SAMPLE_TARGET = $(SRC_SAMPLE:$(SAMPLE)%=$(BIN)%)
SRC_UNITTEST = $(wildcard $(UNITTEST)/*.cpp)
UNITTEST_TARGET = $(SRC_UNITTEST:$(UNITTEST)%=$(BIN)%)

spgemm: rmat $(SAMPLE_TARGET:.cpp=_hw)

unittest: rmat $(UNITTEST_TARGET:.cpp=_hw)

$(BIN)/%_hw: $(SAMPLE)/%.cpp
	mkdir -p $(BIN)
	$(COMPILER) $(FLAGS) $(INCLUDE) -o $@ $^ -DCPP ${TOCOMPILE} ${LIBS}
#	rm -r GenLayout.cpp.*
$(BIN)/%_hw: $(UNITTEST)/%.cpp
	mkdir -p $(BIN)
	$(COMPILER) $(FLAGS) $(INCLUDE) -o $@ $^ -DCPP ${TOCOMPILE} ${LIBS}
		
clean:
	(cd GTgraph; make clean; cd ../..)
	rm -rf ./bin/*
