import numpy as np
import linearalgebra as la

def gaussseidel(A,b,x0 = None, iter=1000):
    _, cols = A.shape
    x = np.copy(x0)
    if x0 is None: x = np.zeros(cols)
    for _ in range(iter):
        for i in range(cols):
            x[i] = (b[i] - A[i,:i] @ x[:i] - A[i,i+1:] @ x[i+1:]) / A[i,i]
    return x

def gaussseidel_all(A,b,x0 = None, iter=1000):
    _, cols = A.shape
    x = np.copy(x0)
    if x0 is None: x = np.zeros(cols)
    xs = [x]
    for _ in range(iter):
        for i in range(cols):
            x[i] = (b[i] - A[i,:i] @ x[:i] - A[i,i+1:] @ x[i+1:]) / A[i,i]
        xs.append(x)
    return xs

def gaussseidel_tridiag(A,b,x0=None,iter=1000):
    _, cols = A.shape
    x = np.copy(x0)
    if x0 is None: x = np.zeros(cols)
    for _ in range(iter):
        x[0] = (b[0] - A[0,1] * x[1]) / A[0,0]
        for i in range(1,cols-1):
            x[i] = (b[i] - A[i,i-1] * x[i-1] - A[i,i+1] * x[i+1]) / A[i,i]
        x[-1] = (b[-1] - A[-1,-2] * x[-2]) / A[-1,-1]
    return x

A = np.array([[16.,3.],[7.,-11.]])
b = np.array([11.,13.])
x0 = np.array([1., 1.])
print(gaussseidel(A,b,x0))

A = la.tridiag(-1, 4, -1, 5000, 5000)
b = np.ones(5000)
gaussseidel_tridiag(A,b)