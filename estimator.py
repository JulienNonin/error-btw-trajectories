#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import validator as validator
from utils import *
from matplotlib.ticker import MaxNLocator


# In[ ]:


dist, dist_to_line = distance, distance_to_line

def find_next_focus_point3(p1, p2, q1, q2):
    """
    distance between p2 (or q1) and the line (p1q1) 
    """
    return dist_to_line(*(p1, q1), p2) < dist_to_line(*(p1, q1), q2)

def isin(elt, L):
    return np.any([np.all(elt == l) for l in L])


# In[ ]:


def sort_points_over_seg(A, B, points):
    """
    TODO : doctring & unittest
    (Points containes points on segment [A, B])
    Add points on segment [A, B] in the right order
    """
    return sorted(points, key = lambda p : distance(p, A))


# In[ ]:


def add_intersections(A, B):
    """TODO : docstring & unittest"""
    S = []
    I = []
    for i in range(len(A)-1):
        line = []
        S.append(A[i])
        for j in range(0, len(B)-1):
#             print(A[i], A[i+1], B[j], B[j+1])
            intersect = intersection_btw_segs(A[i], A[i+1], B[j], B[j+1])
            if np.all(intersect != np.inf):
                line.append(intersect)
                I.append(intersect)
        S.extend(sort_points_over_seg(A[i], A[i+1], line))
    S.append(A[-1])
    return np.array(S), np.array(I)


# In[ ]:


X = np.array([[0, 1], [1, 3], [2, 2], [3, -1], [4, 2], [3, 3]])
Y = np.array([[0, 2], [1, 2], [1.5, 3], [2.5, 3], [2, 0], [1, 0], [0.5, 3]])
# X  = np.array([[0. ,2.], [2., 2.], [2., 0.]])
# Y = np.array([[1.,  1. ],  [2.4, 1. ],  [3.,  0. ]])
plt.plot(*X.T, '--o')
plt.plot(*Y.T, '--o')
S, I = add_intersections(X, Y)
plt.figure()
# print(S)
plt.plot(*S.T, '--o')
plt.plot(*Y.T, '--o')
plt.show()
print(I)


# In[ ]:


def error_btw_trajectories(A, B):
    assert len(A) >= 2 and len(B) >= 1, "IncorrectInputTrajectories"
    S, I  = add_intersections(A, B)
    T, I = add_intersections(B, A)
    I = I if len(I) else []
    i, j = 0, 0
    error = [0]
    
    plt.figure()
    plt.plot(*S.transpose(), '-o', *T.transpose(), '-o')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid()
#     plt.axis("equal")
    
    while i < len(S) - 1 and j < len(T) - 1:
#         print(i, j, f"si={S[i]}, tj={T[j]}, si+1={S[i+1]}, tj+1={T[j+1]}, err={error}")
#         print(isin(T, I), isin(S, I))
        if np.all(S[i] == T[j]):
#             print("=== Both on intersection ===")
            sign = 1 if error[-1] < 0 else -1
            error.append(sign * triangle_area(S[i], T[j+1], S[i+1], ax))
            i += 1
            j += 1
#             print(i, j, len(S), len(T))
#             print(i, j, f"si={S[i]}, tj={T[j]}, si+1={S[i+1]}, tj+1={T[j+1]}, err={error}")

        elif isin(S[i], I):
#             print("=== Intersection on S ===")
            while not isin(T[j], I):
                error[-1] += triangle_area_oriented(S[i], T[j], T[j+1], ax)
                j += 1
        elif isin(T[j], I):
#             print("=== Intersection on T ===", "tj = ", T[j], I, T[j] in I, type(T), type(I))
            while not isin(S[i], I):
                error[-1] += triangle_area_oriented(S[i], T[j], S[i+1], ax)
                i += 1
        else:
            if find_next_focus_point3(S[i], S[i+1], T[j], T[j+1]):
                error[-1] += triangle_area_oriented(S[i], T[j], S[i+1], ax)
                i += 1
            else:
                error[-1] += triangle_area_oriented(S[i], T[j], T[j+1], ax)
                j += 1
    
#     print("=== END phase ===")
    error.append(0)
    if not(i == len(S)-1 and j == len(T)-1):
        if i == len(S) - 1:
            error[-1] += triangle_area_oriented(S[i], T[j], T[j+1], ax)
            j += 1
            for j in range(j, len(T) - 1):
                error[-1] += triangle_area_oriented(S[i], T[j], T[j+1], ax)
        elif j == len(T) - 1: #else ?
            error[-1] += triangle_area_oriented(S[i], T[j], S[i+1], ax)
            i += 1
            for i in range(i, len(S)-1):
                error[-1] += triangle_area_oriented(S[i], T[j], S[i+1], ax)
    
    plt.show()
    Slen = sum(norm(S[i+1] - S[i]) for i in range(len(S) - 1))
    return sum(abs(err) for err in error) / Slen


# In[ ]:


# validator.test(error_btw_trajectories, samples_range = slice(0, None), dirname = "../indoor-location-oracles/Oracles/SampleTrajectories/")
# "../indoor-location-oracles/Oracles/SampleTrajectories/"
# "../indoor-location-oracles/perso/"
