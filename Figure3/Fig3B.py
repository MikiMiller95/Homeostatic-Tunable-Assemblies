import numpy as np


def plot_plasticity_rule(ax, ft, lt, b):
    """Plot the scaled excitatory and inhibitory plasticity rules."""
    # Express the postsynaptic excitatory rate in normalized plotting units.
    r = np.linspace(0.0, 2, 200)
    r1 = 1
    b = 1.0

    # Evaluate and plot the two plasticity rules.
    w_ee_dt = r * (r - b)
    w_ie_dt = r1 * (r - b)
    ax.plot(r, w_ee_dt, color='tab:red', label=r'$\frac{dw_{EE}}{dt}$', linewidth=lt)
    ax.plot(r, w_ie_dt, color='tab:blue', label=r'$\frac{dw_{EI}}{dt}$', linewidth=lt)

    # Format the panel without changing the original plotted ranges or labels.
    ax.legend(loc='upper left', fontsize=ft, frameon=False)
    ax.set_xlabel('FR $r_E$ (Hz)', fontsize=ft)
    ax.set_ylabel(r'$\frac{d w}{d t}$', fontsize=ft)
    ax.annotate('LTD', fontsize=14, color='tab:gray', xy=(0.2, 0.1))
    ax.annotate('LTP', fontsize=14, color='tab:gray', xy=(1.1, 1.32))
    ax.axvline(b, linestyle=':', color='k', linewidth=lt)
    ax.axhline(0, color='k', linewidth=lt)
    ax.set_title(r'$b=20$', fontsize=ft)
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.5, 2)
    ax.set_yticks([0, 1, 2], [0, 1, 2], fontsize=ft)
    ax.set_xticks([0, 2], [0, 40], fontsize=ft)
    return 0
