import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.optimize import minimize_scalar

# Set professional style
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

# Experimental data
experimental_probs = np.array([
    [6, 4, 7, 2, 7, 4, 1],  # Face 1
    [6, 4, 1, 7, 5, 5, 8],  # Face 2
    [2, 1, 5, 4, 2, 4, 0],  # Face 3
    [2, 5, 5, 4, 1, 2, 4],  # Face 4
    [4, 5, 3, 7, 7, 4, 11], # Face 5
    [5, 6, 4, 1, 3, 6, 1]   # Face 6
]) / 25

experimental_error = 0.05
f_values = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0])  # From fair to highly truncated

# Gibbs distribution model
def gibbs_probs(f, beta):
    E = np.array([
        f / 2, f / 2,  # Faces 1 & 2
        0.5, 0.5,      # Faces 3 & 4
        0.5, 0.5       # Faces 5 & 6
    ])
    Z = np.sum(np.exp(-beta * E))
    return np.exp(-beta * E) / Z

# Fit beta using least-squares to minimize error
def fit_beta(f, experimental):
    def loss(beta):
        model = gibbs_probs(f, beta)
        return np.sum((model - experimental) ** 2)
    res = minimize_scalar(loss, bounds=(0.01, 20), method='bounded')
    return res.x, gibbs_probs(f, res.x)

# Fit all f-values
fitted_betas = []
fitted_probs = []
for i, f in enumerate(f_values):
    beta, probs = fit_beta(f, experimental_probs[:, i])
    fitted_betas.append(beta)
    fitted_probs.append(probs)
fitted_probs = np.array(fitted_probs)

# Create figure
fig, ax = plt.subplots(figsize=(6.5, 4))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(f_values)))

# Plot parameters
face_labels = ['Face 1', 'Face 2', 'Face 3', 'Face 4', 'Face 5', 'Face 6']
x = np.arange(len(face_labels))
width = 0.3

# Plot experimental and Gibbs-fitted theoretical data
for i, f in enumerate(f_values):
    offset = width * (i - len(f_values)/2) / len(f_values)

    # Experimental data
    ax.errorbar(x + offset, experimental_probs[:, i], yerr=experimental_error,
                fmt='o', color=colors[i], markersize=5, capsize=2, capthick=1,
                elinewidth=1, label=f'Exp, $f$={f:.1f}')
    
    # Gibbs model prediction
    ax.plot(x + offset, fitted_probs[i], 's',
            markerfacecolor='none', markeredgecolor=colors[i],
            markersize=5, markeredgewidth=1, label=f'Gibbs, $f$={f:.1f}, $\\beta$={fitted_betas[i]:.2f}')

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(face_labels)
ax.set_ylabel('Probability', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
ax.set_axisbelow(True)

# Legend
marker_legend = [
    Line2D([0], [0], marker='o', color='w', label='Experimental',
           markerfacecolor='k', markersize=6),
    Line2D([0], [0], marker='s', color='w', label='Gibbs fit',
           markerfacecolor='none', markeredgecolor='k', markersize=6)
]
ax.legend(handles=marker_legend, loc='upper right', framealpha=1)

# Colorbar
norm = mpl.colors.Normalize(vmin=f_values.min(), vmax=f_values.max())
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=40)
cbar.set_label('Truncation factor $f$', fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout(pad=1.0)
plt.show()
