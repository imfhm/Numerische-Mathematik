import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# ---------- Räuber-Beute-Gleichungen ----------

def my_fun(y, e1, e2, g1, g2):
    return np.array([y[0]*(e1 - g1*y[1]), -y[1]*(e2 - g2*y[0])])

eps1 = 1.1
eps2 = 0.4
gamma1 = 0.4
gamma2 = 0.1
t0, t1 = 0, 100
t = np.linspace(t0, t1, 1000)
y0 = [10, 10]
y = np.zeros((len(t), len(y0)))
y[0, :] = y0
my_integrator = integrate.ode(lambda t, y: my_fun(y,eps1,eps2,gamma1,gamma2)).set_integrator("dopri5")

my_integrator.set_initial_value(y0, t0)
for i in range(1, t.size):
    y[i, :] = my_integrator.integrate(t[i])
    if not my_integrator.successful():
        raise RuntimeError("Could not integrate")

plt.figure(1)
plt.plot(t, y[:,0],label='Beute')
plt.plot(t, y[:,1],label='Räuber')

plt.figure(2)
plt.plot(y[:,0],y[:,1])


plt.figure(1)
plt.legend()
plt.xlabel('Zeit')
plt.ylabel('# Individuen')
plt.figure(2)
plt.xlabel('# Beute')
plt.ylabel('# Räuber')
plt.show()
