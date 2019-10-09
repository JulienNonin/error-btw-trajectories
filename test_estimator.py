import unittest
import numpy as np
from os import listdir
from estimator import *

def fetch_test_data(dirname, correct = True, criterion = '.txt', sep = ',', samples = slice(0, None)):
    """
    >>> fetch_test_data("indoor-location-oracles/Oracles/CorrectInputTrajectories/")[0]
    ['10_parallelTrajectories.txt', array([[1., 1.],
           [3., 1.]]), array([[1., 2.],
           [3., 2.]]), 1.0, 0.001]
    >>> fetch_test_data("indoor-location-oracles/Oracles/IncorrectInputTrajectories/", False)[2]
    [array([[0., 0.],
           [1., 0.]]), array([], dtype=float64)]
    >>> fetch_test_data("indoor-location-oracles/Oracles/IncorrectInputTrajectories/", False)[1]
    [array([[0., 1.]]), array([], dtype=float64)]
    """
    test_file_names = [filename for filename in listdir(dirname) if criterion in filename]
    tests_sample = []
    for filename in sorted(test_file_names)[samples]:
        with open(dirname + filename, "r") as file:
            lines = file.read().splitlines() # getting rid of \n
            data = [[float(n) for n in line.split(sep) if n] for line in lines] # parse data
        
        reference = np.array(list(zip(data[0], data[1]))) # line 0 : x-axis of the reference trajectory, line 1 : y-axis
        acquired = np.array(list(zip(data[2], data[3])))  # line 2 : x-axis of the acquired trajectory, line 3 : y-axis
        if correct:
            expected_output, = data[4] or [-1]
            epsilon, = data[5] or [-1]
            tests_sample.append([filename, reference, acquired, expected_output, epsilon])
        else:
            tests_sample.append([reference, acquired])
    return tests_sample

    
class TestEstimator(unittest.TestCase):
    
    def test_incorrect_input(self):
        dirname = "../indoor-location-oracles/Oracles/IncorrectInputTrajectories/"
        for reference, acquired in fetch_test_data(dirname, correct = False):
            with self.assertRaises(AssertionError):
                error_btw_trajectories(reference, acquired)
    
    def test_correct_input(self):
        dirname = "../indoor-location-oracles/Oracles/CorrectInputTrajectories/"
        for filename, reference, acquired, expected_output, epsilon \
         in fetch_test_data(dirname):
            output = error_btw_trajectories(reference, acquired)

            with self.subTest(test = filename, output = output, expected_output = expected_output):
                self.assertLessEqual(abs(expected_output - output), epsilon)
                
    def test_samples(self):
        dirname = "../indoor-location-oracles/Oracles/SampleTrajectories/"
        for filename, reference, acquired, expected_output, epsilon \
         in fetch_test_data(dirname):
            output = error_btw_trajectories(reference, acquired)

            with self.subTest(test = filename, output = output, expected_output = expected_output):
                self.assertLessEqual(abs(expected_output - output), epsilon)
                
if __name__ == "__main__":
    doctest.testmod()