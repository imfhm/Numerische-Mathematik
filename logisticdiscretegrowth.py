import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(1,4,1000)

alist, Ulist = [], []
for ai in a:
    U = 0.5
    for n in range(1,2000):
        U = ai * U * (1 - U)
        if n >= 1000: 
            alist.append(ai)
            Ulist.append(U)

plt.plot(alist, Ulist, ls='',marker=',')
plt.show()