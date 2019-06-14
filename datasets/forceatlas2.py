
import numpy as np
import time
import os
import argparse
from apgl.graph import VertexList
from scipy.io import mmread, mminfo
import networkx as nx 
import scipy.sparse.csgraph as csgraph
from fa2 import ForceAtlas2

def fa2Layout(G, iteration = 500, isBarnesHut = False, filename="test"):
    forceatlas2 = ForceAtlas2(
                          # Behavior alternatives
                          outboundAttractionDistribution=True,  # Dissuade hubs
                          linLogMode=False,  # NOT IMPLEMENTED
                          adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                          edgeWeightInfluence=1.0,

                          # Performance
                          jitterTolerance=1.0,  # Tolerance
                          barnesHutOptimize=isBarnesHut,
                          barnesHutTheta=1.2,
                          multiThreaded=False,  # NOT IMPLEMENTED

                          # Tuning
                          scalingRatio=2.0,
                          strongGravityMode=False,
                          gravity=1.0,

                          # Log
                          verbose=True)
    start = time.time()
    positionsfa2 = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=iteration)
    end = time.time()
    print('Time required for', iteration, 'iterations in forceatlas2:', end - start)
    outputfile = "fa2output/" + filename.split("/")[1] + "FA2_iter_"+ str(iteration)+"_BH_"+ str(isBarnesHut) + ".txt"
    posfa2 = np.array([list(positionsfa2[item]) for item in positionsfa2])
    output = open(outputfile, "w")
    for i in range(len(posfa2)):
        output.write(str(posfa2[i][0]) + "\t" + str(posfa2[i][1]) + "\n")
    output.close()
    return end - start

def main(filename, iterations=1):
    graph = mmread(filename)
    G = nx.Graph()
    print(mminfo(filename))
    for i in range(int(mminfo(filename)[0])):
        G.add_node(i)
    for i,j in zip(*graph.nonzero()):
        if i > j:
            G.add_edge(i, j)
    posFA2b = fa2Layout(G, iterations, True, filename)
    posFA2 = fa2Layout(G, iterations, False, filename)
    return posFA2b,posFA2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forceatlas2 pipeline...', add_help=True)
    parser.add_argument('-f', '--f', required=True, type=str, help='Filename of graph in mtx format.')
    parser.add_argument('-i', '--i', required=True, type=int, help='Number of iterations.')
    args = parser.parse_args()
    ffile = args.f
    iteration = args.i
    if os.path.isfile(ffile):
        bb,bn = main(ffile, iteration)
        print("Time:", bb, bn)
    else:
        print("File not found!")
