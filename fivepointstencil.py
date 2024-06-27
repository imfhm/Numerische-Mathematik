# 5-Point-Stencil for solving Lu=1
import numpy as np
import matplotlib.pyplot as plt
import linearalgebra as la

def Ah(m, l, h):
    M1 = (l-1) * (m-1)
    Ah = np.zeros((M1, M1))
    for k0 in range(m-1):
        for k1 in range(m-1):
            for i in range(l-1):
                for j in range(l-1):
                    if i == j:
                        if k0 == k1:
                            Ah[i + k0 * (l-1), j + k1 * (l-1)] = -4
                        elif k0 == k1 - 1 or k0 == k1 + 1:
                            Ah[i + k0 * (l-1), j + k1 * (l-1)] = 1
                    elif (i == j - 1 or i == j + 1) and k0 == k1:
                        Ah[i + k0 * (l-1), j + k1 * (l-1)] = 1
    Ah /= h**2
    return Ah
"""
errors = []

for N in [10, 20, 40, 80]:
    l = m = N
    M1 = (l-1)*(m-1)
    Ah = la.tridiag(-1,4,-1,M1,M1)  
    qh = np.ones(M1)
    x0 = np.zeros(M1)
    
    xk_gaussseidel0 = np.zeros(M1)
    error = []
    print("Calculating errors ...")
    z = 0

    xk_gaussseidel = np.zeros(M1)
    for k in range(1000):
        for i in range(M1):
            sum1 = 0
            
            if i % (l-1) != 0: 
                sum1 += -xk_gaussseidel[i-1]
            if i >= l-1: 
                sum1 += -xk_gaussseidel[i - l + 1]
            if i < M1 -1 and (i+1) % (l-1) != 0: 
                sum1 += -xk_gaussseidel[i+1]
            if i <= M1 - l: 
                sum1 += -xk_gaussseidel[i+l-1]
            sum1 *= N**2
            xk_gaussseidel[i] = (qh[i] - sum1) / (4 * N**2)


    for k in range(1000):
        xk_jacobi = np.copy(x0)
        for i in range(M1):
            sum0 = 0
            sum1 = 0
            
            if i % (l-1) != 0: 
                sum0 += -x0[i-1]
                sum1 += -xk_gaussseidel0[i-1]
            if i >= l-1: 
                sum0 += -x0[i - l + 1]
                sum1 += -xk_gaussseidel0[i - l + 1]
            if i < M1 -1 and (i+1) % (l-1) != 0: 
                sum0 += -x0[i+1]
                sum1 += -xk_gaussseidel0[i+1]
            if i <= M1 - l: 
                sum0 += -x0[i+l-1]
                sum1 += -xk_gaussseidel0[i+l-1]
            sum0 *= N**2
            sum1 *= N**2
            xk_jacobi[i] = (qh[i] - sum0) / (4 * N**2)
            xk_gaussseidel0[i] = (qh[i] - sum1) / (4 * N**2)
        x0 = xk_jacobi
        error.append(np.linalg.norm(x0 - xk_gaussseidel))
        if (k > z):
            print(str(k/10) + " percent ...")
            z += 10
    errors.append(error)

plt.semilogy(errors[0], label = "N = " + str(10))
plt.semilogy(errors[1], label = "N = " + str(20))
plt.semilogy(errors[2], label = "N = " + str(40))
plt.semilogy(errors[3], label = "N = " + str(80))
plt.legend()
plt.show()

#ax = plt.figure().add_subplot(projection="3d")
#x = np.linspace(0, 1, N-1)
#y = np.linspace(0, 1, N-1)
#xs, ys = np.meshgrid(x, y)
#ax.plot(xs.flatten(), ys.flatten(), xk_gaussseidel)
#plt.show()"""

