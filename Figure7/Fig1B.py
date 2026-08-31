# Plot the rate-based excitatory and inhibitory plasticity rules used in Figure 1.

import numpy as np


def plot_plasticity_rule(ax, ft, lt, b):
    """Plot the rate-based excitatory and inhibitory plasticity rules."""

    # Evaluate both plasticity rules across postsynaptic firing rates.
    r = np.linspace(0.0, 2, 200)
    r1 = 1
    b = 1.0  # The published panel fixes the threshold at this value.
    w_ee_dt = r * (r - b)
    w_ie_dt = r1 * (r - b)

    # Plot the plasticity curves and mark the threshold.
    ax.plot(r, w_ee_dt, color='tab:red', label=r'$\frac{dw_{ee}}{dt}$', linewidth=lt)
    ax.plot(r, w_ie_dt, color='tab:blue', label=r'$\frac{dw_{ei}}{dt}$', linewidth=lt)
    ax.legend(loc='upper left', fontsize=ft - 2)
    ax.annotate('$b$', fontsize=ft, xy=(b + 0.01, 0.3))
    ax.set_xlabel('Post Synaptic FR $r_E$ (Hz)', fontsize=ft)
    ax.set_ylabel(r'$\dot{w} = \frac{\Delta w}{\Delta t}$', fontsize=ft)
    ax.annotate('LTD', fontsize=24, color='tab:gray', xy=(0.2, 0.0))
    ax.annotate('LTP', fontsize=24, color='tab:gray', xy=(1.0, 1.3))
    ax.axvline(b, linestyle=':', color='k', linewidth=lt)

    # Shade the depression and potentiation regions.
    ax.fill_between(r, -0.5, 0, where=(w_ee_dt <= 0), alpha=0.4)
    ax.fill_between(r, 0, 2, where=(w_ee_dt >= 0), alpha=0.4)

    # Match the limits, ticks, and title used in the figure panel.
    ax.set_title('Rate-based Plasticity Rules', fontsize=ft)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, 2)
    ax.set_yticklabels([-0.5, 0, 0.5, 1, 1.5, 2], fontsize=ft - 1)
    ax.set_xticks([0, 1, 2], [0, 15, 25], fontsize=ft - 1)
    return 0
