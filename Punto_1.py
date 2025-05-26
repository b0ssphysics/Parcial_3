import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# Style settings
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
    'savefig.dpi': 300,
    'figure.autolayout': True
})

# Function: theoretical model
def face_probabilities(f):
    denom = 2 + 4*f/(1 + f**2)
    P1 = P2 = 1 / denom
    P3 = P4 = P5 = P6 = (f / (1 + f**2)) / denom
    return np.array([P1, P2, P3, P4, P5, P6])

# Truncation factors
f_values = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
n_faces = 6
n_trials = 1500  # for Monte Carlo

# Experimental data (original)
original_experimental_probs = np.array([
    [6, 4, 7, 2, 7, 4, 1],
    [6, 4, 1, 7, 5, 5, 8],
    [2, 1, 5, 4, 2, 4, 0],
    [2, 5, 5, 4, 1, 2, 4],
    [4, 5, 3, 7, 7, 4, 11],
    [5, 6, 4, 1, 3, 6, 1]
]) / 25
original_error = 0.05

# Theoretical predictions
theoretical_probs = np.array([face_probabilities(f) for f in f_values])

# Monte Carlo simulation
simulated_probs = np.zeros((n_faces, len(f_values)))
simulated_errors = np.zeros_like(simulated_probs)
rng = np.random.default_rng(42)

for i, f in enumerate(f_values):
    probs = face_probabilities(f)
    outcomes = rng.choice(np.arange(6), size=n_trials, p=probs)
    counts = np.bincount(outcomes, minlength=6)
    simulated_probs[:, i] = counts / n_trials
    simulated_errors[:, i] = np.sqrt(probs * (1 - probs) / n_trials)  # Binomial error

# Plotting
fig, ax = plt.subplots(figsize=(6.5, 4))

face_labels = [f'Face {i+1}' for i in range(6)]
x = np.arange(n_faces)
width = 0.08
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.9, len(f_values)))

for i, f in enumerate(f_values):
    offset = (i - len(f_values)/2) * width + width/2
    x_shifted = x + offset

    # Original experimental data: filled circles with full color
    ax.errorbar(x_shifted, original_experimental_probs[:, i], yerr=original_error,
                fmt='o', color=colors[i], markersize=5,
                capsize=2, elinewidth=0.8, capthick=0.8, label=None)

    # Simulated data: white-filled triangles with colored edge
    ax.errorbar(x_shifted, simulated_probs[:, i], yerr=simulated_errors[:, i],
                fmt='^', markerfacecolor='white', markeredgecolor=colors[i], color=colors[i],
                markersize=5, capsize=2, elinewidth=0.8, capthick=0.8, label=None)

    # Theoretical predictions: hollow squares with colored edge
    ax.plot(x_shifted, theoretical_probs[i], 's',
            markerfacecolor='none', markeredgecolor=colors[i],
            markersize=5, markeredgewidth=1, linestyle='None')

# Grid and axes
ax.set_xticks(x)
ax.set_xticklabels(face_labels)
ax.set_ylabel('Probability')
ax.set_ylim(0, 0.5)
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

# Colorbar
norm = mpl.colors.Normalize(vmin=f_values.min(), vmax=f_values.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=40)
cbar.set_label('Truncation factor $f$', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Custom legend
legend_elements = [
    Line2D([0], [0], marker='o', color='gray', label='Original Experimental', markersize=5, linestyle='None'),
    Line2D([0], [0], marker='^', markerfacecolor='white', markeredgecolor='gray',
           color='gray', label='Simulated (Monte Carlo)', markersize=5, linestyle='None'),
    Line2D([0], [0], marker='s', color='gray', label='Theoretical',
           markerfacecolor='none', markeredgecolor='gray', markersize=5, linestyle='None')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False)

plt.tight_layout()
# Save if needed
# plt.savefig("die_model_comparison.pdf", bbox_inches='tight', pad_inches=0.01)
plt.show()
