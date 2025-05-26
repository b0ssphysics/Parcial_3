import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

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
f_values = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0])  # Now goes from fair die to truncated

# Updated theoretical model
def face_probabilities(f):
    P12 = 1
    P3456 = f
    denom = 2 * P12 + 4 * P3456
    P1 = P2 = P12 / denom
    P3 = P4 = P5 = P6 = P3456 / denom
    return np.array([P1, P2, P3, P4, P5, P6])

theoretical_probs = np.array([face_probabilities(f) for f in f_values])

# Create figure
fig, ax = plt.subplots(figsize=(6.5, 4))

# Use a perceptually uniform colormap
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(f_values)))

# Plot parameters
face_labels = ['Face 1', 'Face 2', 'Face 3', 'Face 4', 'Face 5', 'Face 6']
x = np.arange(len(face_labels))
width = 0.3

# Plot experimental and theoretical data
for i, f in enumerate(f_values):
    offset = width * (i - len(f_values)/2) / len(f_values)

    # Experimental data
    ax.errorbar(x + offset, experimental_probs[:,i], yerr=experimental_error,
                fmt='o', color=colors[i], markersize=5, capsize=2, capthick=1,
                elinewidth=1, label=f'Exp, $f$={f:.1f}')
    
    # Theoretical predictions
    ax.plot(x + offset, theoretical_probs[i,:], 's',
            markerfacecolor='none', markeredgecolor=colors[i],
            markersize=5, markeredgewidth=1, label=f'Theo, $f$={f:.1f}')

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(face_labels)
ax.set_ylabel('Probability', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.5)
ax.set_axisbelow(True)

# Custom legend for markers only
marker_legend = [
    Line2D([0], [0], marker='o', color='w', label='Experimental',
           markerfacecolor='k', markersize=6),
    Line2D([0], [0], marker='s', color='w', label='Theoretical',
           markerfacecolor='none', markeredgecolor='k', markersize=6)
]
ax.legend(handles=marker_legend, loc='upper right', framealpha=1)

# Colorbar for f-values
norm = mpl.colors.Normalize(vmin=f_values.min(), vmax=f_values.max())
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, aspect=40)
cbar.set_label('Truncation factor $f$', fontsize=9)
cbar.ax.tick_params(labelsize=8)

plt.tight_layout(pad=1.0)

# Save if needed
# plt.savefig('die_probabilities.pdf', bbox_inches='tight', pad_inches=0.01)
plt.show()
