import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

def MassSpring_with_force(t, state, f):
    """Simple 1DOF dynamics model: m ddx(t) + k x(t) = f(t)"""
    x, xd = state  # Unpack the state vector
    k = 2.5  # Spring constant
    m = 1.5  # Mass
    xdd = (-k*x + f) / m  # Compute acceleration
    return [xd, xdd]

def force(t):
    """External excitation force"""
    f0 = 1  # Amplitude
    freq = 20  # Frequency
    omega = 2 * np.pi * freq  # Angular frequency
    return f0 * np.sin(omega*t)

# Time range
t_start = 0
t_final = 1

# Set up the solver
state_ode_f = ode(MassSpring_with_force)
state_ode_f.set_integrator('dopri5', rtol=1e-4, nsteps=500,
                          first_step=1e-6, max_step=1e-1, verbosity=True)

# Initial conditions and parameters
state2 = [0.0, 0.0]  # Position and velocity
state_ode_f.set_initial_value(state2, 0)
state_ode_f.set_f_params(force(0))

# Integrate the system
sol = np.array([[t_start, state2[0], state2[1]]], dtype=float)
while state_ode_f.successful() and state_ode_f.t < t_final:
    state_ode_f.set_f_params(force(state_ode_f.t))
    state_ode_f.integrate(t_final, step=True)
    sol = np.append(sol, [[state_ode_f.t, state_ode_f.y[0], state_ode_f.y[1]]], axis=0)

# Plot the results
plt.plot(sol[:, 0], sol[:, 1], label='Position')
plt.plot(sol[:, 0], sol[:, 2], label='Velocity')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
