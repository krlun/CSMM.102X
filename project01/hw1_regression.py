import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")


## Solution for Part 1
def part1(X, y, l):
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    _, n = np.shape(X)
    wRR = np.matmul(np.matmul(np.linalg.inv((l*np.identity(n) + np.matmul(X.T, X))), X.T), y)
    return wRR

wRR = part1(X_train, y_train, lambda_input)  # Assuming wRR is returned from the function
# np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file
np.savetxt('wRR_{}.csv'.format(lambda_input), wRR, delimiter='\n')


## Solution for Part 2
def part2(X, X0, s2, l):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    active = list()
    m, _ = np.shape(X0)
    indices = list(range(m))

    for i in range(10):
        S = calculate_covariance_matrix(X, l, s2) # technically only need to update in regards to the last insertion, see lecture 5 slide 16
        s02 = s2 + np.matmul(X0, np.matmul(S, X0.T))
        row_max = np.argmax(np.diagonal(s02))
        active.append(indices[row_max])
        indices.pop(row_max)
        X = np.vstack((X, X0[row_max]))
        X0 = np.delete(X0, row_max, axis=0)

    return [i + 1 for i in active]

def calculate_covariance_matrix(X, l, s2):
    _, n = np.shape(X)
    S = np.linalg.inv(l*np.identity(n) + (1/s2)*np.matmul(X.T, X))
    return S


active = part2(X_train, X_test, sigma2_input, lambda_input)  # Assuming active is returned from the function
# np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file
np.savetxt('active_{}_{}.csv'.format(lambda_input, int(sigma2_input)), [active], fmt='%d', delimiter=',')