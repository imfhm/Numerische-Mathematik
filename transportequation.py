import numpy as np
import matplotlib.pyplot as plt

# following numerical methods are for solving u_t + cu_x = f
def euler(u0, u1, u2, h, t, n, c, f):
    lam = t / h 
    return u1 - lam * c * (u2 - u0)/2 + f(u1, n * t)*t

def lax_friedrichs(u0, u1, u2, h, t, n, c, f):
    lam = t / h
    return (u0 + u2)/2 - lam*c*(u2-u0)/2 + f(u1, n * t)*t

def upwind(u0, u1, u2, h, t, n, c, f):
    lam = t / h
    return u1 + lam*abs(c)*(u2-2*u1+u0)/2- lam*c*(u2-u0)/2 + f(u1, n * t)*t

def solve(c, f, x0, xe, t0, te, h, t, method):
    xs = np.arange(x0,xe+h,h)
    ts = np.arange(t0,te+h,t)
    M,N = len(xs),len(ts)
    u = np.zeros((N,M))
    u[0]=np.cos(np.pi * xs)
    for i in range(N-1):
        uc = u[i]
        un = np.zeros_like(uc)
        if c > 0: 
            un[0] = -1
            un[-1] = euler(uc[-2], uc[-1], uc[-1], h, t, i, 2*c, f)
        elif c < 0: 
            un[0] = euler(uc[0], uc[0], uc[1],h, t, i, 2*c, f)
            un[-1] = -1
        for j in range(1, len(xs)-1):
            un[j] = method(uc[j-1],uc[j],uc[j+1],h,t,i,c,f)
        u[i+1]=un
    xs, ts = np.meshgrid(xs, ts)
    #return xs.flatten(), ts.flatten(), u.flatten()
    return xs, ts, u

fig = plt.figure(figsize=plt.figaspect(0.3))
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax1.set_zlim(-1,1)
ax2.set_zlim(-1,1)
ax3.set_zlim(-1,1)

xs, ts, u = solve(0.5,(lambda x, t: 0),-1,1,0,2,0.1,0.1, euler)
ax1.plot_surface(xs, ts, u,label="euler")
xs, ts, u = solve(0.5,(lambda x, t: 0),-1,1,0,2,0.005,0.01, lax_friedrichs)
ax2.plot_surface(xs, ts, u,label="lax-friedrichs")
xs, ts, u = solve(0.5,(lambda x, t: 0),-1,1,0,2,0.1,0.1, upwind)
ax3.plot_surface(xs, ts, u,label="upwind")
#ax1.legend()
#ax2.legend()
#ax3.legend()
plt.show()