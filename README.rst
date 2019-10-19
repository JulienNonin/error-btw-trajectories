.. image:: https://circleci.com/gh/JulienNonin/error-btw-trajectories.svg?style=svg&circle-token=14b1663270bb2afa7be91f106967902e3a123617
    :target: https://circleci.com/gh/JulienNonin/error-btw-trajectories

Algorithm for estimating errors between two trajectories
==========================================================

We are interested in **computing metrics that characterise the accuracy of an indoor location system**. Such systems can build upon a wide diversity of technologies (WiFi, Bluetooth, Beacons, Ultra-Wide Bands, Camera, etc.), but it remains unclear how accurate are the locations reported by systems compared to the ground truth (the exact location of a user or an asset). This project therefore aims at developing an algorithm that produces an error estimate for a given indoor location system **by computing the distance between two trajectories**: the one reported by the indoor location system and the ground truth one. The ground truth trajectory is defined as a **sequence of locations `<x,y>`** that represent the ideal trajectory that should be reported by the indoor location system. Given two sequences of locations—whose lengths can differ—the algorithm should report a mean error estimates as the sum of the areas formed by the two trajectories and then divided by the length of the ground truth trajectory.


How to contribute?
-------------------
see the naming conventions_

.. _conventions : https://github.com/JulienNonin/error-btw-trajectories/docs