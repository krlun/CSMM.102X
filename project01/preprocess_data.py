import numpy as np
import sys, os

def load_data(infile):
    data = list()
    with open(infile, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            data.append(line.rstrip().split(','))
    return data

def preprocess(data):
    X_train = data[1:100]
    y_train = list()
    X_test = data[100:]

    for i in range(len(X_train)):
        y_train.append(X_train[i][-1])
        X_train[i][-1] = str(1)
    
    for i in range(len(X_test)):
        X_test[i][-1] = str(1)

    return X_train, y_train, X_test

def write_data(outfile, data):
    if os.path.exists(outfile):
        os.remove(outfile)
    
    f = open(outfile, 'a+')
    for line in data:
        f.write(','.join(line) + '\n')
    f.close()

def write_data_y(outfile, data):
    if os.path.exists(outfile):
        os.remove(outfile)

    f = open(outfile, 'a+')
    for line in data:
        f.write(line + '\n')
    f.close()

def main(argv):
    data = load_data(argv[0])
    X_train, y_train, X_test = preprocess(data)
    write_data('X_train.csv', X_train)
    write_data_y('y_train.csv', y_train)
    write_data('X_test.csv', X_test)


if __name__ == "__main__":
    main(sys.argv[1:])