import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Imagine in a cable there is an initial potential.
# It will propogate to left and right with in each case lower amplitude.

# Initial displacement
def phi(x):
    if -np.pi / 2 <= x <= np.pi / 2: return np.cos(x)
    return 0

# Initial rate of change
def psi(x):
    return 0

c = 0.1
xs = np.linspace(-10, 10, 100)
us = []

N = 100
for t in range(N):
    u = []
    for x in xs:
        u.append(0.5 * (phi(x + c * t) + phi(x - c * t)) + 1/(2*c) * integrate.quad(psi, x - c * t, x + c * t)[0])
    us.append(u)

fig = plt.figure()
ax = fig.add_subplot()
line = ax.plot(xs, us[0])[0]

def update(frame):
    line.set_ydata(us[frame])
    return line

ani = animation.FuncAnimation(fig=fig,func=update,frames=N,interval=1/60.)
plt.show()