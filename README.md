<<<<<<< HEAD
# flGenForDiGraph

## Fast Layout Generation for Force-Directed Graph

This tool generates layout of force-directed graph using many-cores which is generally faster than equivalent sequential algorithm. We have implemented the popular Spring-Electrical model. We have also provided a jupyter notebook to visualize the graph in 2D plot.

## System Requirements

Users need to have following softwares/tools installed in their PC.
```
GCC version >= 4.9
OpenMP version >= 4.5
```

## Compile and Run

To check unit test result, use the following command:
```
$ make clean
$ make unittest
$ ./bin/unittest_hw ./datasets/input/3elt_dual.mtx 1 500
```
Here, 1 is the number of threads and 500 is the number of iterations. An output file  will be generated in 'datasets/output' folder where final coordinates (layout) of all nodes will be reported. To draw graph jump to graph drawing below.

## Graph Drawing and Metrics

It is required that following python libraries are installed on your pc:
```
pip install apgl scipy networkx s_gd2 fa2
```

Open the jupyter notebook named 'drawGraph.ipynb' located in datasets folder (If you do not have jupyter notebook installed in your pc, please install it). Give appropriate matrix market file name (.mtx) in line 'graph = mmread("input/3elt\_dual.mtx")' that were used to generate the layout. Here, "3elt\_dual.mtx" file was used to generate layout. After running each line in jupyter notebook, a layout graph with aesthetic metrics' values will be shown in the end. Please also modify iterations that were used to run code.

=======
# ParallelGraphVis

# Test
>>>>>>> 862fa7b9c54875b587dad33922236619b87772c5
