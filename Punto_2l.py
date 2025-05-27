import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
G, M = 1.0, 1.0      # Gravitational constant and mass
eps = 0.01           # Softening parameter to avoid singularities
a, c = 1.0, 0.8      # Fixed axis values

def grad_potential(x, y, z, a, b, c, eps):
    denom = (x**2 / a**2 + y**2 / b**2 + z**2 / c**2 + eps**2)**1.5
    dPhi_dx = G * M * x / (a**2 * denom)
    dPhi_dy = G * M * y / (b**2 * denom)
    dPhi_dz = G * M * z / (c**2 * denom)
    return -np.array([dPhi_dx, dPhi_dy, dPhi_dz])

def full_dynamics(t, Y, a, b, c, eps):
    x, y, z, px, py, pz = Y
    dxdt = px
    dydt = py
    dzdt = pz
    dpxdt, dpydt, dpzdt = grad_potential(x, y, z, a, b, c, eps)
    return [dxdt, dydt, dzdt, dpxdt, dpydt, dpzdt]

def lyapunov_dynamics(t, Y, a, b, c, eps):
    state = Y[:6]
    delta = Y[6:]
    derivs = np.array(full_dynamics(t, state, a, b, c, eps))
    derivs_perturbed = np.array(full_dynamics(t, state + delta, a, b, c, eps))
    delta_dot = derivs_perturbed - derivs  # 6 elements
    return np.concatenate([derivs, delta_dot])

# Initial conditions
Y0 = [1.0, 0.0, 0.0, 0.0, 0.6, 0.0]     # Position and momentum
delta0 = 1e-8 * np.ones(6)             # Small deviation vector
IC = np.concatenate([Y0, delta0])

# Time span
t_span = (0, 500)
t_eval = np.linspace(*t_span, 5000)

# Vary b/a ratio
axis_ratios = [1.0, 0.9, 0.8, 0.7]
lyapunov_curves = []

for b in axis_ratios:
    print(f"Simulating for b/a = {b:.2f}")
    sol = solve_ivp(lyapunov_dynamics, t_span, IC, args=(a, b, c, eps),
                    method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-9)

    delta = sol.y[6:]
    delta_norm = np.linalg.norm(delta, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        lyap_exp = np.log(delta_norm / delta_norm[0]) / sol.t
        lyap_exp[np.isnan(lyap_exp)] = 0.0

    lyapunov_curves.append((b, sol.t, lyap_exp))

# Plotting
plt.figure(figsize=(10, 6))
for b, t_vals, lyap_exp in lyapunov_curves:
    plt.plot(t_vals, lyap_exp, label=f'b/a = {b}')
plt.xlabel('Time')
plt.ylabel('Lyapunov Exponent Estimate')
plt.title('MLE Estimate vs Time for Varying Axis Ratios')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
