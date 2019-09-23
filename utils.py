#!/usr/bin/env python
# coding: utf-8

# # Utils

# In[2]:


import numpy as np
import doctest
from matplotlib.patches import Polygon


# ## Distance

# In[23]:


def distance(P, Q):
    """
    >>> distance(*np.array([[-8, 9],[0, 0]]))
    12.041594578792296
    >>> distance(np.array([10, 20]), np.array([-4, 5]))
    20.518284528683193
    >>> distance(np.array([-3, 5]), np.array([-3, 5]))
    0.0
    """
    return np.sqrt(np.sum((P-Q)**2 ))

def distance_to_line(A1, A2, B):
    """ distance from B to line (A1A2)"""
    # equation of the line (A1A2) is defined as by = ax + c
    b, a = A2 - A1
    c = b*A1[1] - a*A1[0]
    return abs(b*B[1] - a*B[0] - c)/np.hypot(a, b)

if __name__ == "__main__":
    doctest.testmod()


# ## Vectors

# In[26]:


def norm(X):
    return distance(X, 0)

def orientation(A, B, C):
    """ Returns:
        *  1 if A-B-C are counterclockwise ordered
        *  0              colinear
        * -1             clockwise ordered
    """
    r = (B[1] - A[1]) * (C[0] - B[0]) - (C[1] - B[1]) * (B[0] - A[0])
    if np.abs(r) < 1e-5 : # r == 0
        return 0
    elif r > 0:
        return 1
    else:
        return -1
    
def is_colinear(v1, v2):
    return abs(np.dot(v1, v2)) == norm(v1) * norm(v2)


# ## Intersections

# In[28]:


def find_eq_line(A, B):
    """ax + by = c"""
    a = B[1] - A[1]
    b = A[0] - B[0]
    c = b*A[1] + a*A[0]
    return a, b, c


# ### Finding the intersection point of two lines

# In[9]:


def intersection_btw_lines(p1, p2, q1, q2):
    a1, b1, c1 = find_eq_line(p1, p2)
    a2, b2, c2 = find_eq_line(q1, q2)
    denom = a1*b2 - a2*b1
    xs = (c1*b2 - c2*b1) / denom if denom else np.inf
    ys = (a1*c2 - a2*c1) / denom if denom else np.inf
    return np.array([xs, ys])


# ### Finding the intersection point of two line segments
# Let $p_1$ and $p_2$ be the (2D) endpoints of one segment and let $q_1$ and $q_2$ be the endpoints of the other. A parametrization of these lines are defined as:
# $$
# \left\{\begin{array}{l}{p_{1}+t_p\left(p_{2}-p_{1}\right)} \\ {p_{3}+t_q\left(p_{4}-p_{3}\right)}\end{array}\right.
# $$
# where $t_p, t_q \in [0,1]$. Thus, the segments intersect iff there exists $(s,t)$ such that:
# $$p_1+t_p(p_2-p_1) = q_1+t_q(q_2-q_1)$$
# i.e.
# $$t_q(q_2-q_1) + t_p(p_1-p_2) = p_1 - q_1$$
# We can define our system using matrices ($p_1$, $p_2$, $q_1$, $q_2$ being a column vector of size 2) :
# $$
# \underbrace{\left[
#     \begin{array}{ll}
#         q_2-q_1 & p_1-p_1\end{array}
# \right]}_A \times 
# \underbrace{\left[
#     \begin{array}{l}
#         t_q \\ t_p\end{array}
# \right]}_T = 
# \underbrace{\left[
#     \begin{array}{ll}
#         p_1 & q_1\end{array}
# \right]}_B
# $$
# * if a solution $(t_p, t_q)$ exists and is in $[0,1]\times[0,1]$, the segments intersect (at $p_{1}+t_p(p_{2}-p_{1})$).
# * if A is not inversible, the segments have the same slope (we need to test if segments are colinear or parallel)

# In[11]:


def intersection_btw_segs(p1, p2, q1, q2, display = False):
    """ Find intersection point between [p1, p2] and [q1, q2] 
    Parameters :
        - p1, p2, q1, q2 : np.ndarrays of shape (2,)
    Returns : the coordinates of the intersection points,  
        if the line segments intersect
    
    [TODO] Manage the case where the matrix is singular
    
    >>> intersection_btw_segs(*np.array([[10, 5], [8, 1], [3,7],[0,1]]))
    array([inf, inf])
    """
    
    a = np.array([q2-q1, p1-p2]).transpose()
    if display:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')
        plt.plot([q1[0], q2[0]], [q1[1], q2[1]], '-o')
    if np.linalg.det(a): # if a is invertible (if lines intersect)
        b = p1 - q1
        t = np.linalg.solve(a,b)
        if np.all(0 < t) and np.all(t < 1): # segments intersect
            intersection = p1 + t[1]*(p2-p1)
            if display:
                plt.plot(*(intersection), 'o')
            return intersection 
    return np.array([np.inf, np.inf])

if __name__=="__main__":
    doctest.testmod()


# ### Finding the intersection between a line and a line segment

# In[22]:


def intersection_btw_line_seg(p1, p2, q1, q2):
    """intersection between (p1, p2) and ]q1, q2["""
    s = intersection_btw_lines(p1, p2, q1, q2)
    # we have two find whether or not s belongs to ]q1, q2]
    ks = np.dot(q2-q1, s-q1)
    kp = np.dot(q2-q1, q2-q1)
    if 0 < ks < kp:
        return s
    elif ks >= kp:
        return np.inf
    else: # ks <= 0
        return -np.inf


# ## Triangle area

# In[29]:


NTRIANGLE = 0
def triangle_area(A, B, C, ax = None):
    global NTRIANGLE
    NTRIANGLE += 1
    if ax != None:
        ax.add_patch(Polygon([A, B, C], facecolor=["grey", "lightgrey"][NTRIANGLE % 2], ec = "black", alpha = 0.3))
    return 0.5 * abs((B[0]-A[0])*(C[1]-A[1]) - (C[0]-A[0])*(B[1]-A[1]))

def triangle_area_oriented(A, B, C, ax = None):
    area = 0.5 * abs((B[0]-A[0])*(C[1]-A[1]) - (C[0]-A[0])*(B[1]-A[1])) * orientation(A, B, C)
    color = "yellow" if area < 0 else "lightblue"
    if ax != None:
        ax.add_patch(Polygon([A, B, C], facecolor=color, ec = "black", alpha = 0.4))
    return area