#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from matplotlib.ticker import MaxNLocator

def test(estimator, epsilon = .00001, samples_range = slice(0, None), display = False):
    """Tests whether the function 'estimator' correctly estimates the
    difference between two trajectories. A battery of tests contained
    in the test folder is used for this purpose. It raises an `Assertion-
    Error` if any test fails.
    Parameters :
        - estimator: function that takes 2 required 2D np.ndarrays,
            returns a float
        - epsilon: float, used as a tolerance threshold between the
            expected value and the value returned by the function estimator
        - slices_range : a slicing object to select the tests. e.g.:
            * slice(3): select only the test n°3
            * slice(0, None): select all tests (by default)
            * slice(0,4): select the first 4 tests
            * slice(3, None): select from the 3rd
        - display: boolean - if True plots the trajectories
    Returns : True if all tests have been validated, False if any has failed
    """
    dirname = "test/" # directory where the test files are located
    validated = True # boolean returned by the function
    for filename in sorted(listdir(dirname)[samples_range]):
        if "test" in filename: # for each test
            print(filename, end="\t")
            
            # Fetching data
            path = dirname + filename
            X = np.loadtxt(path, skiprows=0, max_rows=2, unpack=True)
            Y = np.loadtxt(path, skiprows=2, max_rows=2, unpack=True)
            res_true = np.loadtxt(path, skiprows=4)
            
            # Compute the error estimate algorithm
            res = estimator(X, Y)
            
            # Plotting trajectories
            if display:
                plt.figure()
            #    plt.clf()
                plt.plot(*X.transpose(), '--o', *Y.transpose(), '--o')
                plt.axis("equal")
                plt.grid()
                plt.title(filename)
            
            # Comparing the expected value with the estimation computes by
            # the estimator, with a tolerance of +/- epsilon
            if res_true - epsilon <= res <= res_true + epsilon:
                print(f"ok \t The expected value is indeed {res}")
            else :
                print(f"NOT ok \t The expected value is {res_true}, but the output value is {res}.")
                validated = False
    return validated

if __name__ == "__main__":
    estimate = lambda X, Y : np.abs(np.trapz(*X.transpose()[::-1]) - np.trapz(*Y.transpose()[::-1]))
    test(estimate)