# Algorithm for estimating errors between two trajectories

We are interested in **computing metrics that characterise the accuracy of an indoor location system**. Such systems can build upon a wide diversity of technologies (WiFi, Bluetooth, Beacons, Ultra-Wide Bands, Camera, etc.), but it remains unclear how accurate are the locations reported by systems compared to the ground truth (the exact location of a user or an asset). This project therefore aims at developing an algorithm that produces an error estimate for a given indoor location system **by computing the distance between two trajectories**: the one reported by the indoor location system and the ground truth one. The ground truth trajectory is defined as a **sequence of locations `<x,y>`** that represent the ideal trajectory that should be reported by the indoor location system. Given two sequences of locations—whose lengths can differ—the algorithm should report a mean error estimates as the sum of the areas formed by the two trajectories and then divided by the length of the ground truth trajectory.


## Testing the error estimate algorithm
> :warning: At the moment the results are not normalized by the length of the real trajectory.
### Test samples
A battery of tests is located in the folder `test`. Each file  `[test<i>]<description>.txt` represents a test sample, with has the format shown in the following example:
```python
0	1	2	3	4	# abscissa of the points of the real trajectory
0	0	1	1	0	# ordinate of the points of the real trajectory
0	1	2	3	4	# abscissa of the points reported by the location system
3	3	2	2	5	# ordinate of the points reported by the location system
9					# expected result
```

Here is an example of code that allows you to retrieve the data from such a test file:
```python
import numpy as np
filename = "test/[test1]simple.txt"
real_trajectory = np.loadtxt(filename, skiprows=0, max_rows=2, unpack=True)
reported_trajectory = np.loadtxt(filename, skiprows=2, max_rows=2, unpack=True)
expected_value = np.loadtxt(filename, skiprows=4)
```

### Use `validator.py`
To test a function `estimator_to_be_tested` on all tests, you can use the file `validator.py` and its function `test`:
```python
import validator as validator
validator.test(stimator_to_be_tested)
```

This function tests whether the function `estimator_to_be_tested` correctly estimates the difference between two trajectories, using a battery of tests. It raises an `AssertionError` if any test fails.