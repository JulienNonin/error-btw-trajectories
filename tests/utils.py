import trajectories_error.estimator as estimator

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