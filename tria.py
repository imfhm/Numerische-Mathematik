import matplotlib.pyplot as plt
from numpy import *

def pos_orient(a,b,c):
    return (a[0]*b[1]+a[1]*c[0]+b[0]*c[1]-c[0]*b[1]-c[1]*a[0]-b[0]*a[1])>0

def tria(A):
    p = argsort(A[:,0])
    
    coh=[p[0],p[1],p[2],p[0]]
    if not pos_orient(A[p[0],:],A[p[1],:],A[p[2],:]):
        coh=[p[0],p[2],p[1],p[0]]
    T=[coh[:-1]]

    for m in range(3,len(p)):
        u=0
        while(u < len(coh)-1):   
            pu0 = A[coh[u],:]
            pm = A[p[m],:]
            pu1 = A[coh[u+1],:]

            if pos_orient(pu0,pm,pu1):
                T.append([coh[u],p[m],coh[u+1]])
                
                if coh[u-1]==p[m]: 
                    coh.pop(u)
                    u-=1
                else: 
                    coh.insert(u+1,p[m])
                    u+=1
            u+=1            
    return T

c=cos(1)
A=array([[cos(k),cos(c*k)] for k in range(50)])
#A=array([[0,5],[1,0],[2,-1],[2,2],[4,5]])
T=tria(A)
plt.triplot(A[:,0],A[:,1],T,'o-')
plt.show()
