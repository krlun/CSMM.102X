import numpy as np
import sys

from numpy.core.fromnumeric import shape

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
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    indices = list()

    for i in range(10):
        S = calculate_covariance_matrix(X_train, lambda_input, sigma2_input)
        s02 = sigma2_input**2 + np.matmul(X_test, np.matmul(S, X_test.T))
        row_max = np.argmax(np.diagonal(s02))
        indices.append(row_max)





def calculate_covariance_matrix(X, l, s):
    _, n = np.shape(X)
    S = np.linalg.inv(l*np.identity(n) + (1/s**2)*np.matmul(X.T, X))
    return S


active = part2()  # Assuming active is returned from the function
# np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file
