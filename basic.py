import numpy as np
import matplotlib.pyplot as plt
import timeit
import matplotlib.animation as animation

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

# comparison of the above gauss implementations and the one in numpy.linalg.solve times are not compared as numpy.linalg.solve uses LU decomposition with partial pivoting (only row interchanges)  
def compare_gauss_numpy():
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

def compare_prothero_robinson(lam, g, gprime, dt, steps):
    for x0 in range(-10, 10):
        xs = [x0]
        ts = np.arange(0, dt*steps + dt/2, dt)
        for i in range(steps):
            xs.append(xs[-1] + dt * (lam * (xs[-1] - g(ts[i])) + gprime(ts[i])))
        plt.plot(ts, xs, label="Explicit Euler Method", color="red")

        xs=[x0]
        for i in range(steps):
            xs.append((xs[-1] + dt*(lam * -g(ts[i+1]) + gprime(ts[i+1])))/(1-lam*dt))
        plt.plot(ts, xs, label="Implicit Euler Method", color="blue")
    #plt.legend()
    plt.show()

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

# Recursive Evaluation of the discretization of the transport equation
def r(j,n):
    if n == 0: return "u"+str(j)+"0"
    return r(j,n-1)+"+k*("+r(j+1,n-1)+"-"+r(j-1,n-1)+")"+"+f"#+str(j)+str(n-1)+"*t"

def VMOL_animation():
    N = 1000
    h = 0.4 / N
    c = 0.5

    
    fig, ax = plt.subplots()
    t = np.linspace(0,3,40)
    g = -9.81
    v0 = 12
    z = g * t ** 2 / 2 + v0 * t

    v02 = 5
    z2 = g * t ** 2 / 2 + v02 * t

    scat = ax.scatter(t[0],z[0],c="b",s=5,label=f'v0={v0} m/s')
    line2 = ax.plot(t[0],z2[0],label=f'v0 = {v02} m/s')[0]
    ax.set(xlim=[0,3], ylim=[-4,10],xlabel='Time [s]', ylabel = 'Z [m]')
    ax.legend()

    def update(frame):
        x = t[:frame]
        y = z[:frame]
        data = np.stack([x,y]).T
        scat.set_offsets(data)
        line2.set_xdata(t[:frame])
        line2.set_ydata(z2[:frame])
        return (scat, line2)
    
    ani = animation.FuncAnimation(fig=fig,func=update,frames=40,interval=30)
    plt.show()

VMOL_animation()

"""
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
"""
