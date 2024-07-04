from scipy.integrate import solve_bvp
import numpy as np
import matplotlib.pyplot as plt
M = 100
N = 100
c = 0.5
t = 0.1 / N

x_span = (0, 1)
x_eval = np.linspace(*x_span, M)
  
u = lambda x: np.sin(np.pi * x) + 5 * np.sin(3 * np.pi * x)

figure = plt.figure()
ax = figure.add_subplot(projection='3d')  
ax.plot(np.zeros(M), x_eval, [u(x) for x in x_eval])

for i in range(N):
    def ode(x, y):
        y1, y2 = y
        return [y2, (y1 - u(x)) / (t * c)]
    
    def bc(ya, yb):
        return np.array([ya[0], yb[0]])
    
    sol = solve_bvp(ode, bc, x_eval, np.zeros((2, x_eval.size)))
    u = lambda x: sol.sol(x)[0] 
 
    ax.plot(i * t * np.ones(x_eval.size), x_eval, sol.sol(x_eval)[0])
plt.show()
    