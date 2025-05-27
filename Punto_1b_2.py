import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.optimize import minimize_scalar

# --- Matplotlib styling ---
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
})

# --- Experimental data ---
experimental_probs = np.array([
    [6, 4, 7, 2, 7, 4, 1],
    [6, 4, 1, 7, 5, 5, 8],
    [2, 1, 5, 4, 2, 4, 0],
    [2, 5, 5, 4, 1, 2, 4],
    [4, 5, 3, 7, 7, 4, 11],
    [5, 6, 4, 1, 3, 6, 1]
]) / 25

experimental_error = 0.05
f_values = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0])
face_labels = ['Face 1', 'Face 2', 'Face 3', 'Face 4', 'Face 5', 'Face 6']
x = np.arange(len(face_labels))
width = 0.3
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(f_values)))

# --- Gibbs model ---
def gibbs_probs(f, beta):
    E = np.array([f/2, f/2, 0.5, 0.5, 0.5, 0.5])
    Z = np.sum(np.exp(-beta * E))
    return np.exp(-beta * E) / Z

def fit_beta(f, experimental):
    def loss(beta):
        model = gibbs_probs(f, beta)
        return np.sum((model - experimental) ** 2)
    res = minimize_scalar(loss, bounds=(0.01, 20), method='bounded')
    return res.x, gibbs_probs(f, res.x)

# --- Fit model ---
fitted_betas = []
fitted_probs = []
residuals = []

for i, f in enumerate(f_values):
    beta, probs = fit_beta(f, experimental_probs[:, i])
    fitted_betas.append(beta)
    fitted_probs.append(probs)
    residuals.append(probs - experimental_probs[:, i])

fitted_probs = np.array(fitted_probs)
residuals = np.array(residuals)

# --- Create main figure and axes ---
fig, (ax1, ax2) = plt.subplots(
    nrows=2, figsize=(7, 6.5), sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'right': 0.88}
)

# --- Top panel: Probabilities ---
for i, f in enumerate(f_values):
    offset = width * (i - len(f_values)/2) / len(f_values)
    ax1.errorbar(x + offset, experimental_probs[:, i], yerr=experimental_error,
                 fmt='o', color=colors[i], markersize=5, capsize=2)
    ax1.plot(x + offset, fitted_probs[i], 's', markerfacecolor='none',
             markeredgecolor=colors[i], markersize=5, markeredgewidth=1)

ax1.set_ylabel('Probability')
ax1.set_xticks(x)
ax1.set_xticklabels(face_labels)
ax1.grid(True, linestyle=':', alpha=0.4)
ax1.set_axisbelow(True)

# Custom legend
custom_handles = [
    Line2D([0], [0], marker='o', color='k', label='Experimental', linestyle=''),
    Line2D([0], [0], marker='s', color='k', label='Gibbs fit', linestyle='', markerfacecolor='none')
]
ax1.legend(handles=custom_handles, loc='upper right', framealpha=1)

# --- Bottom panel: Residuals ---
for i, f in enumerate(f_values):
    offset = width * (i - len(f_values)/2) / len(f_values)
    ax2.bar(x + offset, residuals[i], width=width/len(f_values), color=colors[i], alpha=0.8)

ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_ylabel('Residuals')
ax2.set_xticks(x)
ax2.set_xticklabels(face_labels)
ax2.grid(True, linestyle=':', alpha=0.4)
ax2.set_axisbelow(True)

# --- Proper right-side colorbar ---
norm = mpl.colors.Normalize(vmin=f_values.min(), vmax=f_values.max())
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])

# Create new axis just for the colorbar
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Truncation factor $f$', fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.subplots_adjust(left=0.1, right=0.88, hspace=0.15)
plt.show()
