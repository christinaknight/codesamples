import sys
import networkx as nx
import pandas as pd
import numpy as np
import scipy.special as sc
import math
import matplotlib.pyplot as plt
import time

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def compute(infile, outfile):
    file = open(infile)
    type(file)
    D = pd.read_csv(infile)
    vars = D.columns.tolist()
    file.close()
    G = K2Search(vars, D)
    print(bayesian_score(vars, G, D))
    write_gph(G, vars, outfile)
    nx.draw(G, with_labels = True)
    plt.savefig(outfile + ".png")

def sub2ind(siz, x):
    return np.ravel_multi_index(x, siz)

def statistics(vars, G, D):
    r = np.array(list((D.max())))
    D = D.to_numpy() - 1
    n = len(vars)
    q = [math.prod([r[j] for j in G.predecessors(i)]) for i in range(n)]
    M = [np.zeros((q[i], r[i])) for i in range(n)]
    for s in range(D.shape[0]):
        sample = D[s,:]
        for i in range(n):
            k = sample[i]
            parents = np.array(list(G.predecessors(i)))
            j = 0
            if len(parents) != 0:
                j = sub2ind(r[parents], np.array(sample)[parents])
            M[i][j,k] += 1.0
    return M

def bayesian_score_component(M, prior):
    p = np.sum(sc.loggamma(prior + M))
    p -= np.sum(sc.loggamma(prior))
    p += np.sum(sc.loggamma(np.sum(prior, axis=1)))
    p -= np.sum(sc.loggamma(np.sum(prior, axis=1) + np.sum(M, axis=1)))
    return p

def make_prior(vars, G, D):
    r = np.array(list((D.max())))
    n = len(vars)
    q = [math.prod([r[j] for j in G.predecessors(i)]) for i in range(n)]
    return [np.ones((q[i], r[i])) for i in range(n)]

def bayesian_score(vars, G, D):
    n = len(vars)
    M = statistics(vars, G, D)
    prior = make_prior(vars, G, D)
    return sum(bayesian_score_component(M[i], prior[i]) for i in range(n))

def K2Search(vars, D):
    G = nx.DiGraph()
    G.add_nodes_from(range(len(vars)))
    nodes = G.nodes()
    for count, node in enumerate(nodes):
        y = bayesian_score(vars, G, D)
        while True:
            y_best, j_best = -math.inf, 0
            for j in range(count):
                if G.has_edge(j, node) == False:
                    G.add_edge(j, node)
                    y_prime = bayesian_score(vars, G, D)
                    if y_prime > y_best:
                        y_best, j_best = y_prime, j
                    G.remove_edge(j, node)
            if y_best > y:
                y = y_best
                G.add_edge(j_best, node)
            else:
                break
    return G

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    start_time = time.time()
    compute(inputfilename, outputfilename)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
