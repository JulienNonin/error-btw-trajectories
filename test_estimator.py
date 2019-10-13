import unittest
import numpy as np
from os import listdir
import os
from estimator import *
from unittest.mock import patch # prevent ploting figures

def fetch_test_data(dirname, correct = True, criterion = '.txt', sep = ',', samples = slice(0, None)):
    """
    >>> fetch_test_data("./indoor-location-oracles/Oracles/CorrectInputTrajectories/")[0]
    ['10_parallelTrajectories.txt', Trajectory([Point(1.0, 1.0), Point(3.0, 1.0)]), Trajectory([Point(1.0, 2.0), Point(3.0, 2.0)]), 1.0, 0.001]
    >>> fetch_test_data("./indoor-location-oracles/Oracles/IncorrectInputTrajectories/", False)[2]
    [Trajectory([Point(0.0, 0.0), Point(1.0, 0.0)]), Trajectory([])]
    >>> fetch_test_data("./indoor-location-oracles/Oracles/IncorrectInputTrajectories/", False)[1]
    [Trajectory([Point(0.0, 1.0)]), Trajectory([])]
    """

    test_file_names = [filename for filename in listdir(dirname) if criterion in filename]
    tests_sample = []
    for filename in sorted(test_file_names)[samples]:
        with open(dirname + filename, "r") as file:
            lines = file.read().splitlines() # getting rid of \n
            data = [[float(n) for n in line.split(sep) if n] for line in lines] # parse data
        
        reference_coord = list(zip(data[0], data[1])) # line 0 : x-axis of the reference trajectory, line 1 : y-axis
        acquired_coord = list(zip(data[2], data[3]))  # line 2 : x-axis of the acquired trajectory, line 3 : y-axis
        reference = Trajectory([Point(x, y) for x, y in reference_coord])
        acquired = Trajectory([Point(x, y) for x, y in acquired_coord])
        
        if correct:
            expected_output, = data[4] or [-1]
            epsilon, = data[5] or [-1]
            tests_sample.append([filename, reference, acquired, expected_output, epsilon])
        else:
            tests_sample.append([reference, acquired])
    return tests_sample

    
class TestEstimator(unittest.TestCase):
    
    @patch('matplotlib.pyplot.figure')
    def test_incorrect_input(self, mock_show):
        dirname = "./indoor-location-oracles/Oracles/IncorrectInputTrajectories/"
        for reference, acquired in fetch_test_data(dirname, correct = False):
            with self.assertRaises(AssertionError):
                reference.error_with(acquired)
    
    @patch('matplotlib.pyplot.figure')
    def test_correct_input(self, mock_show):
        dirname = "indoor-location-oracles/Oracles/CorrectInputTrajectories/"
        for filename, reference, acquired, expected_output, epsilon \
         in fetch_test_data(dirname):
            output = reference.error_with(acquired)

            with self.subTest(test = filename, output = output, expected_output = expected_output):
                self.assertLessEqual(abs(expected_output - output), epsilon)
    
    @patch('matplotlib.pyplot.figure')
    def test_samples(self, mock_show):
        dirname = "indoor-location-oracles/Oracles/SampleTrajectories/"
        for filename, reference, acquired, expected_output, epsilon \
         in fetch_test_data(dirname):
            output = reference.error_with(acquired)

            with self.subTest(test = filename, output = output, expected_output = expected_output):
                self.assertLessEqual(abs(expected_output - output), epsilon)