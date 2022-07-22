# -*- coding: utf-8 -*-
"""
Used for the test described in chapter  5.7.1. Plots figure 5.1 at the end
(actually: a plot which is CLOSE to it since execution times will certainly
 differ by using different calculators)

@author: 
    Ninniri Matteo (m.ninniri1@studenti.unipi.it)
    Davide Amadei (d.amadei@studenti.unipi.it)    
"""

import matplotlib.pyplot as plt
import numpy as np

from QR_Regressor import QR_Regressor

###############################################################################
y = []

start = 1000
end = 20000
step = 100
columns = 5
times = 5

r = np.arange(start, end, step = step)

for j in range(times):
    index = 0
    print(j)
    for i in r:
        # if(i % 10 == 0):
        #     print(i)
        
        A = np.random.rand(i, columns)
        
        while(np.linalg.matrix_rank(A) != columns):
            A = np.random.rand(i, columns)
        b = np.random.rand(i, 1)
        
        rg = QR_Regressor()
        rg.thin_qr(A)
    
        if(j == 0):
            y.append(rg.factorization_time)
        else:
            y[index] += rg.factorization_time
        
            
        index += 1
        
y = [o/times for o in y]

plt.figure()
plt.title("Time as m increases")
plt.plot(r, y)
plt.plot([start, end - 1], [y[0], y[-1]])
plt.xlabel("m")
plt.ylabel("Time (seconds)")
plt.show()