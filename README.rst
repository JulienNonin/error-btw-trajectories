.. image:: https://circleci.com/gh/JulienNonin/error-btw-trajectories.svg?style=svg&circle-token=14b1663270bb2afa7be91f106967902e3a123617
    :target: https://circleci.com/gh/JulienNonin/error-btw-trajectories

Algorithm for estimating errors between two trajectories
==========================================================

We are interested in **computing metrics that characterise the accuracy of an
indoor location system**. Such systems can build upon a wide diversity of
technologies (WiFi, Bluetooth, Beacons, Ultra-Wide Bands, Camera, etc.),
but it remains unclear how accurate are the locations reported by systems
compared to the ground truth (the exact location of a user or an asset).
This project therefore aims at developing an algorithm that produces an error
estimate for a given indoor location system **by computing the distance between
two trajectories**: the one reported by the indoor location system and the
ground truth one. The ground truth trajectory is defined as a **sequence of
locations `<x,y>`** that represent the ideal trajectory that should be reported
by the indoor location system. Given two sequences of locations—whose lengths
can differ—the algorithm should report a mean error estimates as the sum of the
areas formed by the two trajectories and then divided by the length of the
ground truth trajectory.

Installation
-------------
First clone the project

.. code-block:: bash

    git clone https://github.com/JulienNonin/error-btw-trajectories.git
    cd error-btw-trajectories

`trajectories_error` runs on Python 3.7. Install Python requirements:

.. code-block:: bash

    pip install -r requirements.txt

A submodule is contained inside the project to share some oracles. You have to
update this submodule in order to access these tests:

.. code-block:: bash

    cd tests/shared-oracles
    git submodule update --init
    git checkout master
    cd ../../

To doctest all functions and methods:

.. code-block:: bash

    python -m doctest trajectories_error/*.py
    python -m doctest tests/*.py

Tests are loacted inside the `tests/` folder. Run `unittest` in order to test
all st oracles:

.. code-block:: bash

    python -m unittest tests.test_estimator

Usage
------
Import the module

.. code-block:: python

    import trajectories_error.estimator as est
    import matplotlib.pyplot as plt

You can define your own trajectories and compute the error between them:

.. code-block:: python

    reference = est.Trajectory([est.Point(0, 0), est.Point(1, 1), est.Point(2, -1)])
    acquired = est.Trajectory([est.Point(0,0), est.Point(1, -1), est.Point(-1, 2)])
    reference.error_with(acquired)

To fetch trajectories from a `.txt` file, and display the two trajectories:

.. code-block:: python

    A, B, expected_error, epsilon = est.fetch_data(path)
    A.error_with(B, display=True)
    plt.show()



How to contribute?
-------------------
see the naming conventions_

.. _conventions : https://github.com/JulienNonin/error-btw-trajectories/blob/master/docs/CONTRIBUTING.md