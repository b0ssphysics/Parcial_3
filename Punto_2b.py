import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Get parameters from user input
def get_float(prompt, default):
    try:
        return float(input(f"{prompt} [default={default}]: ") or default)
    except ValueError:
        return default

def get_int(prompt, default):
    try:
        return int(input(f"{prompt} [default={default}]: ") or default)
    except ValueError:
        return default

def get_span(prompt, default=(0, 40)):
    try:
        raw = input(f"{prompt} (format: start,end) [default={default}]: ") or f"{default[0]},{default[1]}"
        parts = [int(x) for x in raw.strip().split(',')]
        return tuple(parts) if len(parts) == 2 else default
    except:
        return default

# User-defined parameters
base_a = get_float("Enter base parameter a", 1.0)
base_b = get_float("Enter base parameter b", 0.8)
base_c = get_float("Enter base parameter c", 0.6)
n_orbits = get_int("Enter number of orbits", 200)
t_span = get_span("Enter time span", (0, 40))
t_eval = np.linspace(*t_span, 2000)

# Constants
G, M = 1.0, 1.0
eps = 0.01
base_IC = np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.1])
rng = np.random.default_rng(42)

# Create variations in (a, b, c)
param_variants = [
    (base_a - 0.1, base_b, base_c),
    (base_a, base_b + 0.1, base_c),
    (base_a, base_b, base_c + 0.1),
    (base_a + 0.1, base_b - 0.1, base_c - 0.1)
]

# Set up 3D subplots
fig = plt.figure(figsize=(14, 12))
axes = [fig.add_subplot(2, 2, i+1, projection='3d') for i in range(4)]

# Main loop over variants
for idx, (a, b, c) in enumerate(param_variants):
    orbits = []

    def grad_potential(pos):
        x, y, z = pos
        denom = (x**2 / a**2 + y**2 / b**2 + z**2 / c**2 + eps**2)**1.5
        dPhi_dx = G * M * x / (a**2 * denom)
        dPhi_dy = G * M * y / (b**2 * denom)
        dPhi_dz = G * M * z / (c**2 * denom)
        return -np.array([dPhi_dx, dPhi_dy, dPhi_dz])

    def dynamics(t, Y):
        x, y, z, px, py, pz = Y
        dxdt, dydt, dzdt = px, py, pz
        dpxdt, dpydt, dpzdt = grad_potential([x, y, z])
        return np.array([dxdt, dydt, dzdt, dpxdt, dpydt, dpzdt])

    print(f"Integrating orbits for parameters: a={a:.2f}, b={b:.2f}, c={c:.2f}")
    for i in range(n_orbits):
        IC = base_IC + rng.normal(scale=1e-3, size=6)
        try:
            sol = solve_ivp(dynamics, t_span, IC, t_eval=t_eval, rtol=1e-6, atol=1e-9)
            if sol.success:
                orbits.append(sol.y[:3])
        except:
            continue

    ax = axes[idx]
    colors = plt.cm.plasma(np.linspace(0, 1, len(orbits)))
    for i, orbit in enumerate(orbits):
        ax.plot(*orbit, lw=0.6, alpha=0.4, color=colors[i])

    ax.set_title(f"a={a:.2f}, b={b:.2f}, c={c:.2f}", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=30, azim=135)

    # Set equal axis limits
    def set_equal_3d(ax):
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        spans = limits[:, 1] - limits[:, 0]
        centers = np.mean(limits, axis=1)
        max_span = max(spans) / 2
        new_limits = np.array([centers - max_span, centers + max_span]).T
        ax.set_xlim3d(*new_limits[0])
        ax.set_ylim3d(*new_limits[1])
        ax.set_zlim3d(*new_limits[2])

    set_equal_3d(ax)

plt.suptitle("Effect of Small Variations in Potential Parameters on Orbit Chaos", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()
