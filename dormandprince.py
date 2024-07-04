# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=89a13692f6ecd6117e8b608d40e43127bf75736b
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

def dopri(y0, te, f, h, Atol, Rtol, nsteps = 500):
    ys, ts, errors = [y0], [0], [0]
    t = ts[-1]
    arsteps = [0]
    while t < te:
        y = ys[-1]
        t = ts[-1]
        
        err = 2
        fac = 0.9    
        facmax = 10
        facmin = 0.2
        sc = Atol + np.abs(y) * Rtol
        
        steps = 0
        while err > 1 and steps < nsteps:
            k1 = f(t,               y)
            k2 = f(t + h / 5,       y +             k1 * h / 5)
            k3 = f(t + 3 * h / 10,  y + 3 *         k1 * h / 40 + 9 *           k2 * h/ 40)
            k4 = f(t + 4 * h / 5,   y + 44 *        k1 * h / 45 - 56 *          k2 * h/ 15 + 32 *          k3 * h/ 9)
            k5 = f(t + 8 * h / 9,   y + 19372 *     k1 * h / 6561 - 25360 *     k2 * h/ 2187 + 64448 *     k3 * h/ 6561 - 212 *   k4 *h/ 729)
            k6 = f(t + h,           y + 9017 *      k1 * h / 3168 - 355 *       k2 * h/ 33 - 46732 *       k3 * h/ 5247 + 49 *    k4 *h/ 176 - 5103 * k5 * h/ 18656)
            k7 = f(t + h,           y + 35 *        k1 * h / 384 + 500 *                                   k3 * h/ 1113 + 125 *   k4 *h/ 192 - 2187 * k5 * h/ 6784 + 11 * k6 * h/ 84)

            yn = y + h * (35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84)
            z = y + h * (5179 * k1 / 57600 + 7571 * k3 / 16695 + 393 * k4 / 640 - 92097 * k5 / 339200 + 187 * k6 / 2100 + k7 / 40)

            err = np.abs((yn - z) / sc)
            
            h = h * min(facmax, max(facmin, fac / np.power(err, 1. / 5)))
            steps += 1
            
            if err > 1 and steps == nsteps: facmax = 1

        ts.append(t + h)
        ys.append(yn)
        errors.append(err)
        arsteps.append(steps)
    return ts, ys, errors, arsteps

y0 = 0.3
def f(x, y, p):
    if (x + 0.05)**2 + (y+0.15)**2 + p <= 1: return x**2 + 2 * y**2 + p
    return 2 * x **2 + 3 * y**2 - 2 + p
#def f(x, y):
#    return y

p = 0
ts, result1, errors, arsteps = dopri(y0, 1, lambda x, y: f(x,y,p), 1. / 22, 10e-5, 10e-5)
#plt.plot(ts, result1)

ts, result2, errors, arsteps = dopri(y0, 1, lambda x, y: f(x,y,p+0.001), 1. / 22, 10e-5, 10e-5)
plt.plot(ts, np.array(result2) - np.array(result1))
#plt.plot(ts, result2)
plt.show()

"""plt.plot(ts, result, color="b")
#result_accepted = [result[n] for n in range(N) if errors[n] < 10e-5]
#result_rejected = [result[n] for n in range(N) if errors[n] >= 10e-5]
#ts_accepted = [n * h for n in range(N) if errors[n] < 10e-5]
#ts_rejected = [n * h for n in range(N) if errors[n] >= 10e-5]
#plt.scatter(ts_accepted, result_accepted, marker='o')
#plt.scatter(ts_rejected, result_rejected, marker='x')

y0, t0 = 0.3, 0
solver = ode(f)
solver.set_integrator('dopri5', rtol = 10e-5, atol = 10e-5)
solver.set_initial_value(y0, t0)

t = np.linspace(0, 1, 23)
k = 1
sol = [y0]
while solver.successful() and solver.t < 1:
    solver.integrate(t[k])
    sol.append(solver.y)
    k+=1
plt.plot(t, sol, color="r")
plt.show()"""