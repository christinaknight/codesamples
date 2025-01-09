import sys
import numpy as np
import pandas as pd
import time

def compute(infile, outputfilename):
    file = open(infile)
    type(file)
    D = pd.read_csv(infile)
    D = D.to_numpy()
    file.close()
    if infile == "small.csv":
        num_states = 100
        num_actions = 4
        discount = 0.95
    elif infile == "medium.csv":
        num_states = 50000
        num_actions = 7
        discount = 1
    else:
        num_states = 312020
        num_actions = 9
        discount = 0.95
    q_matrix = np.zeros((num_states, num_actions))
    policy = q_learning(q_matrix, D, discount)
    with open(outputfilename, 'w') as f:
        for s in range(len(policy)):
            f.write(str(policy[s]) + "\n")
    return f

def q_learning(q_matrix, data, discount):
    policy = []

    for i in range(100):
        for row in data:
            s = row[0]
            a = row[1]
            r = row[2]
            sp = row[3]
            q_matrix = update(q_matrix, s - 1, a - 1, r, sp - 1, discount)

    print(q_matrix)

    for row in range(q_matrix.shape[0]):
        policy.append(np.argmax(q_matrix[row]) + 1)

    print(q_matrix.shape)
    print(policy)
    return policy

def update(Q, s, a, r , sp, discount):
    Q[s, a] += 0.1 * (r + discount * np.max(Q[sp,:]) - Q[s, a])
    return Q

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv")
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    start_time = time.time()
    compute(inputfilename, outputfilename)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()