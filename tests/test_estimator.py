import unittest
import numpy as np
import os
from tests.utils import fetch_data
from unittest.mock import patch # prevent ploting figures
import trajectories_error.estimator as estimator

current_dir = os.path.dirname(__file__)


def fetch_data(filename, contains_solution=True, criterion='.txt', sep=','):
    """
    >>> fetch_data("tests/shared-oracles/Oracles/CorrectInputTrajectories/10_parallelTrajectories.txt")[1:]
    [Trajectory([Point(1.0, 1.0), Point(3.0, 1.0)]), Trajectory([Point(1.0, 2.0), Point(3.0, 2.0)]), 1.0, 0.001]
    >>> fetch_data("tests/shared-oracles/Oracles/IncorrectInputTrajectories/3_EmptyAcquiredTrajectory.txt", False)[1:]
    [Trajectory([Point(0.0, 0.0), Point(1.0, 0.0)]), Trajectory([])]
    >>> fetch_data("tests/shared-oracles/Oracles/IncorrectInputTrajectories/2_EmptyAcquiredTrajectory.txt", False)[1:]
    [Trajectory([Point(0.0, 1.0)]), Trajectory([])]
    """
    with open(filename, "r") as file:
        lines = file.read().splitlines() # getting rid of \n
        data = [[float(n) for n in line.split(sep) if n] for line in lines] # parse data
        
    reference_coord = list(zip(data[0], data[1])) # line 0 : x-axis of the reference trajectory, line 1 : y-axis
    acquired_coord = list(zip(data[2], data[3]))  # line 2 : x-axis of the acquired trajectory, line 3 : y-axis
    reference = estimator.Trajectory([estimator.Point(x, y) for x, y in reference_coord])
    acquired = estimator.Trajectory([estimator.Point(x, y) for x, y in acquired_coord])

    if contains_solution:
        expected_output, = data[4] or [-1]
        epsilon, = data[5] or [-1]
        return [filename, reference, acquired, expected_output, epsilon]
    else:
        return [filename, reference, acquired]

def fetch_test_data(dirname, contains_solution=True, criterion='.txt', sep=',', samples=slice(0, None)):
    """
    >>> fetch_test_data("shared-oracles/Oracles/CorrectInputTrajectories/")[0][1:]
    [Trajectory([Point(1.0, 1.0), Point(3.0, 1.0)]), Trajectory([Point(1.0, 2.0), Point(3.0, 2.0)]), 1.0, 0.001]
    >>> fetch_test_data("shared-oracles/Oracles/IncorrectInputTrajectories/", False)[2][1:]
    [Trajectory([Point(0.0, 0.0), Point(1.0, 0.0)]), Trajectory([])]
    >>> fetch_test_data("shared-oracles/Oracles/IncorrectInputTrajectories/", False)[1][1:]
    [Trajectory([Point(0.0, 1.0)]), Trajectory([])]
    """
    global current_dir
    dirname = filename = os.path.join(current_dir, dirname)
    test_file_names = [filename for filename in os.listdir(dirname) if criterion in filename]
    tests_sample = []
    for filename in sorted(test_file_names)[samples]:
        tests_sample.append(fetch_data(dirname + filename, contains_solution=contains_solution, criterion=criterion, sep=sep))
    return tests_sample

    
class TestEstimator(unittest.TestCase):
    
    @patch('matplotlib.pyplot.figure')
    def test_incorrect_input(self, mock_show):
        dirname = "shared-oracles/Oracles/IncorrectInputTrajectories/"
        for filename, reference, acquired in fetch_test_data(dirname, contains_solution = False):
            with self.assertRaises(AssertionError):
                reference.error_with(acquired)
    
    @patch('matplotlib.pyplot.figure')
    def test_correct_input(self, mock_show):
        dirname = "shared-oracles/Oracles/CorrectInputTrajectories/"
        for filename, reference, acquired, expected_output, epsilon \
         in fetch_test_data(dirname):
            output = reference.error_with(acquired)

            with self.subTest(test = filename, output = output, expected_output = expected_output):
                self.assertLessEqual(abs(expected_output - output), epsilon)
    
    @patch('matplotlib.pyplot.figure')
    def test_samples(self, mock_show):
        dirname = "shared-oracles/Oracles/SampleTrajectories/"
        for filename, reference, acquired, expected_output, epsilon \
         in fetch_test_data(dirname):
            output = reference.error_with(acquired)

            with self.subTest(test = filename, output = output, expected_output = expected_output):
                self.assertLessEqual(abs(expected_output - output), epsilon)
                
    @patch('matplotlib.pyplot.figure')
    def test_perso_samples(self, mock_show):
        dirname = "perso-tests/"
        for filename, reference, acquired, expected_output, epsilon \
         in fetch_test_data(dirname):
            output = reference.error_with(acquired)

            with self.subTest(test = filename, output = output, expected_output = expected_output):
                self.assertLessEqual(abs(expected_output - output), epsilon)