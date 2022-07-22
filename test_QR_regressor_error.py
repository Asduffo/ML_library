# -*- coding: utf-8 -*-
"""
Runs the experiments in chapter 5.6.2 and plots the figures shown in that chapter

@author: 
    Amadei Davide (d.amadei@studenti.unipi.it)    
    Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import numpy as np

from QR_Regressor import QR_Regressor

A = np.array([
    [1, 1, 1],
    [1, 2, 3],
    [2, 4, 6],
    [4, 8, 12.000002]])
print("rank(A) = ", np.linalg.matrix_rank(A))

A_ready = A.copy()
A_ready = np.hstack((A_ready, np.ones((A_ready.shape[0], 1))))

b = np.array([
    [10],
    [21],
    [41],
    [81.000008]])
x = np.array([
    [2],
    [3],
    [4],
    [1]])

rg = QR_Regressor()
rg.fit(A, b)

np.set_printoptions(precision=64)
print("A = \n", A)
print("b = \n", b)
print("x = \n", x)

x_t = rg.w.T
print("x_t = ", x_t)

print("Ax_t = \n", rg.predict(A))
print("Ax_t - b = \n", rg.predict(A) - b)

norm_Ax_norm_b = np.linalg.norm(np.dot(A_ready, x))/np.linalg.norm(b)
print("norm_Ax_norm_b = ", norm_Ax_norm_b)

r1 = np.dot(rg.Q1.T, rg.predict(A) - b)
print("r1 = \n", r1)

frac = np.linalg.norm(r1)/np.linalg.norm(b)
print("||r1||/||b|| = ", frac)

residual_error = np.linalg.norm(x_t - x)/np.linalg.norm(x)
print("x_t - x = ", x_t - x, "\n||x_t - x|| = ", np.linalg.norm(x_t - x))
print("||x|| = ", np.linalg.norm(x))
print("||x_t - x||/||x|| = ",residual_error)


cond_A = np.linalg.cond(A)
print("condition number of A = ", cond_A)

right_hand_side = (cond_A/norm_Ax_norm_b)*frac
print("K(A)/cos(theta)*||r1||/||b|| = ", right_hand_side)

if (residual_error <= right_hand_side):
    print(residual_error, " <= ", right_hand_side)
else:
    print("The inequality does not hold")
