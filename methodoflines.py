import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import linearalgebra as la
import fivepointstencil as fps
from scipy.integrate import solve_ivp

c = 0.5

l = m = 15
M1 = (l-1) * (m-1)
h = 1. / l

Ah = c*fps.Ah(l, m, h)
f = np.ones(M1)

print(Ah)
def odes(t, y):
    return Ah @ y + f

init_cond = np.zeros(M1)
time_span = (0, 0.4)
N = 1000
solution = solve_ivp(odes, time_span, init_cond, t_eval=np.linspace(*time_span, num=N))

xs = np.linspace(0, 1, l-1)
ys = np.linspace(0, 1, m-1)
xv, yv = np.meshgrid(xs, ys)
xv = xv.flatten()
yv = yv.flatten()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
surf = ax.plot(xv, yv, solution.y[:,0])[0]

def update(frame):
    surf.set_data_3d(xv, yv, solution.y[:,frame])
    return surf

ani = animation.FuncAnimation(fig=fig,func=update,frames=N,interval=200)
plt.show()