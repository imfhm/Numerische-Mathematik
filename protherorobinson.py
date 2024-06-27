import numpy as np
import matplotlib.pyplot as plt

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