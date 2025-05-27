import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
G, M = 1.0, 1.0
a =int(input("Enter the value for a (e.g., 1.0): "))

b =int(input("Enter the value for b (e.g., 1.0): "))

c =int(input("Enter the value for c (e.g., 1.0): "))

eps = 0.01

# Potential gradient
def grad_potential(pos):
    x, y, z = pos
    denom = (x**2 / a**2 + y**2 / b**2 + z**2 / c**2 + eps**2)**1.5
    dPhi_dx = G * M * x / (a**2 * denom)
    dPhi_dy = G * M * y / (b**2 * denom)
    dPhi_dz = G * M * z / (c**2 * denom)
    return -np.array([dPhi_dx, dPhi_dy, dPhi_dz])

# Equations of motion
def dynamics(t, Y):
    x, y, z, px, py, pz = Y
    dxdt, dydt, dzdt = px, py, pz
    dpxdt, dpydt, dpzdt = grad_potential([x, y, z])
    return np.array([dxdt, dydt, dzdt, dpxdt, dpydt, dpzdt])

# Integration settings
n_orbits = int(input("Enter the number of orbits to integrate: "))
t_span = tuple(int(x) for x in input("Enter the integration time span (e.g., 0,400): ").strip("() ").split(","))
# Increased integration time
t_eval = np.linspace(*t_span, 10000)  # Increased temporal resolution

# Base initial condition
base_IC = np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.1])

# Create 4 different initial conditions by small perturbations
perturbations = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.1, 0.05, 0.0, -0.05, 0.0, 0.02]),
    np.array([-0.1, 0.1, 0.0, 0.05, -0.05, -0.01]),
    np.array([0.05, -0.1, 0.0, 0.02, 0.1, 0.0])
]

# Prepare plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, perturb in enumerate(perturbations):
    IC = base_IC + perturb
    sol = solve_ivp(dynamics, t_span, IC, t_eval=t_eval, rtol=1e-6, atol=1e-9)
    
    x_vals = sol.y[0]
    px_vals = sol.y[3]
    
    ax = axes[i]
    ax.plot(x_vals, px_vals, color='green', lw=1)
    ax.set_xlabel(r"$q_i$", fontsize=12)
    ax.set_ylabel(r"$p_i$", fontsize=12)
    ax.set_title(f"Orbit {i+1}: IC = {np.round(IC, 3)}", fontsize=14)
    ax.grid(True)

plt.tight_layout()
plt.show()
