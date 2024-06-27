import numpy as np

# The matrix maximum norm. It is the maximum of all row sums where for every matrix entry the absolute value is taken. 
def norm_matrix_infinity(A):
    return max([sum(abs(A[i,:])) for i in range(len(A))])

# Tridiagonal matrix with a on sub-, b on main- and c on superdiagonal 
def tridiag(a,b,c,n,m):
    A=np.zeros((n,m))
    A[0,0],A[0,1]=b,c
    for i in range(1,len(A)-1):
        A[i,i-1],A[i,i],A[i,i+1]=a,b,c
    A[-1,-2],A[-1,-1]=a,b
    return A

# Recursive Evaluation of the discretization of the transport equation
def r(j,n):
    if n == 0: return "u"+str(j)+"0"
    return r(j,n-1)+"+k*("+r(j+1,n-1)+"-"+r(j-1,n-1)+")"+"+f"#+str(j)+str(n-1)+"*t"