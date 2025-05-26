import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# Styling
plt.style.use('seaborn-paper')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': True,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# Model and inputs
def face_probabilities(f):
    denom = 2 + 4*f/(1 + f**2)
    P1 = P2 = 1 / denom
    P3 = P4 = P5 = P6 = (f / (1 + f**2)) / denom
    return np.array([P1, P2, P3, P4, P5, P6])

f_values = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
n_faces = 6
n_trials = 1500

# Monte Carlo simulation
simulated_probs = np.zeros((n_faces, len(f_values)))
rng = np.random.default_rng(42)

for i, f in enumerate(f_values):
    probs = face_probabilities(f)
    outcomes = rng.choice(np.arange(6), size=n_trials, p=probs)
    counts = np.bincount(outcomes, minlength=6)
    simulated_probs[:, i] = counts / n_trials

# Theoretical values
theoretical_probs = np.array([face_probabilities(f) for f in f_values]).T

# Residuals: MC - Theoretical
residuals = simulated_probs - theoretical_probs

# Plot
fig, ax = plt.subplots(figsize=(6.5, 3))
face_labels = [f'Face {i+1}' for i in range(6)]
x = np.arange(n_faces)
width = 0.08
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(f_values)))

for i, f in enumerate(f_values):
    offset = (i - len(f_values)/2) * width + width/2
    x_shifted = x + offset
    ax.bar(x_shifted, residuals[:, i], width=width, color=colors[i], alpha=0.9)

ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_ylabel('Residuals')
ax.set_xticks(x)
ax.set_xticklabels(face_labels)
ax.grid(True, linestyle=':', alpha=0.5)
ax.set_axisbelow(True)

# Colorbar
norm = mpl.colors.Normalize(vmin=f_values.min(), vmax=f_values.max())
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=40)
cbar.set_label('Truncation factor $f$', fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout()
# plt.savefig("residuals_mc_vs_theory.pdf", bbox_inches='tight')
plt.show()
