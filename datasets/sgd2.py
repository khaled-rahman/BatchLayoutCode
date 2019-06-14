import numpy as np
import time
import os
import argparse
from apgl.graph import VertexList
from scipy.io import mmread, mminfo
import networkx as nx
import scipy.sparse.csgraph as csgraph
import s_gd2, random
import scipy.sparse.csgraph as csgraph

def s_gd2Layout(G, iteration = 500, isBarnesHut = False, filename="test"):
    num_iter = iteration
    epsilon = 0.1
    start0 = time.time()
    d = csgraph.shortest_path(G, directed=True, unweighted=True)
    end0 = time.time()
    n = d.shape[0]
    constraints = []
    w_min = float('inf')
    w_max = 0
    for i in range(n):
        for j in range(i):
            w = 1/d[i,j]**2
            w_min = min(w, w_min)
            w_max = max(w, w_max)
            constraints.append((i,j,w))
    eta_max = 1/w_min
    eta_min = epsilon/w_max
    lambd = np.log(eta_min / eta_max) / (num_iter - 1);
    eta = lambda t: eta_max*np.exp(lambd*t)
    schedule = []
    for i in range(num_iter):
        schedule.append(eta(i))
    positions = np.random.rand(n, 2)
    start = time.time()
    for c in schedule:
        random.shuffle(constraints)
        for i,j,w in constraints:
            wc = w*c
            if (wc > 1):
                wc = 1
            pq = positions[i] - positions[j]
            mag = np.linalg.norm(pq)
            r = (d[i,j] - mag) / 2
            m = wc * r * pq/mag
            positions[i] += m
            positions[j] -= m
        
    end = time.time()
    print('Time required for', iteration, 'iteraions in s_gd2:', end - start + end0 - start0)
    outputfile = "sgd2output/" + filename.split("/")[1] + "SGD2_iter_"+ str(iteration) + ".txt"
    output = open(outputfile, "w")
    for i in range(len(positions)):
        output.write(str(positions[i][0]) + "\t" + str(positions[i][1]) + "\n")
    output.close()
    return end - start + end0 - start0

def main(filename, iterations=1):
    G = mmread(filename)
    possgd2 = s_gd2Layout(G, iterations, False, filename)
    return possgd2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SGD2 pipeline...', add_help=True)
    parser.add_argument('-f', '--f', required=True, type=str, help='Filename of graph in mtx format.')
    parser.add_argument('-i', '--i', required=True, type=int, help='Number of iterations.')
    args = parser.parse_args()
    ffile = args.f
    iteration = args.i
    if os.path.isfile(ffile):
        sgd = main(ffile, iteration)
        print("Time:", sgd)
    else:
        print("File not found!")
