import numpy as np
import matplotlib.pyplot as plt

# simple gauss elimination for square regular matrices
# problems: n³ runtime, roundoff errors
def gauss(A,b):
    for i in range(len(A)):
        for k in range(i+1, len(A)):
            b[k]=b[k]-b[i]*A[k,i]/A[i,i]
            A[k,:]=A[k,:]-A[i,:]*A[k,i]/A[i,i]
    x=np.zeros(len(A))
    x[-1]=b[-1]/A[-1,-1]
    for i in reversed(range(len(A)-1)):
        x[i]=(b[i]-A[i,i+1:]@x[i+1:])/A[i,i]
    return x

# gauss with pivot for square regular matrices  
# problems: n³ runtime
def gauss_pivot(A,b):
    for i in range(len(A)):
        pivot = np.argmax(np.abs(A[i:,i]))
        A[[i,i+pivot]]=A[[i+pivot,i]]
        b[i],b[i+pivot]=b[i+pivot],b[i]
        for k in range(i+1, len(A)):
            b[k]=b[k]-b[i]*A[k,i]/A[i,i]
            A[k,:]=A[k,:]-A[i,:]*A[k,i]/A[i,i]
    x=np.zeros(len(A))
    x[-1]=b[-1]/A[-1,-1]
    for i in reversed(range(len(A)-1)):
        x[i]=(b[i]-A[i,i+1:]@x[i+1:])/A[i,i]
    return x

errors0 = []
errors1 = []
for n in range(1,10):
    A=np.random.normal(0,1,(n,n))
    b=np.random.rand(n)
    np.fill_diagonal(A,1e-14)
    A0 = np.copy(A)
    A1 = np.copy(A)
    b0 = np.copy(b)
    b1 = np.copy(b)
    x0 = gauss(A0,b0)
    x1 = gauss_pivot(A1,b1)
    x = np.linalg.solve(A,b)
    errors0.append(np.linalg.norm(x0-x))
    errors1.append(np.linalg.norm(x1-x))
plt.plot(errors0,label="Gauss")
plt.plot(errors1,label="Gauss with Pivot")
plt.legend()
plt.show()