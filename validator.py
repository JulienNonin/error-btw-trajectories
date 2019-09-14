#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from matplotlib.ticker import MaxNLocator

def test(estimator, epsilon = .00001):
    """Tests whether the function 'estimator' correctly estimates the
    difference between two trajectories. A battery of tests contained
    in the test folder is used for this purpose. It raises an `Assertion-
    Error` if any test fails.
    Parameters :
        - estimator : function that takes 2 required 2D np.ndarrays,
            returns a float
        - epsilon : float, used as a tolerance threshold between the
            expected value and the value returned by the function estimator
    Returns : None
    """
    dirname = "test/" # directory where the test files are located
    for filename in sorted(listdir(dirname)):
        if "[test" in filename: # for each test
            print(filename, end="\t")
            
            # Fetching data
            path = dirname + filename
            X = np.loadtxt(path, skiprows=0, max_rows=2, unpack=True)
            Y = np.loadtxt(path, skiprows=2, max_rows=2, unpack=True)
            res_true = np.loadtxt(path, skiprows=4)
            
            # Compute the error estimate algorithm
            res = estimator(X, Y)
            
            # Plotting trajectories
            plt.clf()
            plt.plot(*X.transpose(), '--o', *Y.transpose(), '--o')
            plt.axis("equal")
            plt.grid()
            
            # Comparing the expected value with the estimation computes by
            # the estimator, with a tolerance of +/- epsilon
            assert res_true - epsilon <= res <= res_true + epsilon, \
                f"The expected value for {filename} is {res_true}, not {res}."
            print("ok")

if __name__ == "__main__":
    estimate = lambda X, Y : np.abs(np.trapz(*X.transpose()[::-1]) - np.trapz(*Y.transpose()[::-1]))
    test(estimate)