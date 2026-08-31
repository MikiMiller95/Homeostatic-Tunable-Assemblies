from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression

from Fig3B import plot_plasticity_rule

# ──────────────────────────────────────────────────────────────────────
#  Plotting style
# ──────────────────────────────────────────────────────────────────────
TITLE_FS = 11
LABEL_FS = 11
TICK_FS = 8
LEGEND_FS = 9

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}"})
dark_tab = {
    'th:blue'   : '#0e3651',
    'th:orange' : '#733906',
    'th:green'  : '#144814',
    'th:red'    : '#601212',
    'th:purple' : '#432e55',
    'th:brown'  : '#3f2722',
    'th:pink'   : '#663657',
    'th:gray'   : '#393939',
    'th:olive'  : '#55550f',
    'th:cyan'   : '#0a565d',
}
ft = LABEL_FS

# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
def scaled_value(raw: float) -> float:
    return raw / factor * tau_rprim


def build_path(kind: str, wee_sc: float, wei_sc: float, seed: int) -> Path:
    fname = (f'{kind}Ne_{N_E}_T{T}_wee{wee_sc}_wei{wei_sc}'
             f'_wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}'
             f'_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}'
             f'_tauwei{tau_wei}_seed{seed}.csv')
    return data_dir / fname


def load_csv_stack(kind_prefix: str, wee_raw: float, wei_raw: float):
    """Load and stack the requested rate or weight CSV files across seeds."""
    wee_sc = scaled_value(wee_raw)
    wei_sc = scaled_value(wei_raw)
    if 'rates' in kind_prefix:
        raw_re, raw_ri = [], []
        for s in seeds:
            p = build_path(kind_prefix, wee_sc, wei_sc, s)
            rates = pd.read_csv(p).values
            raw_re.append(rates[0, :-1])
            raw_ri.append(rates[1, :-1])
        raw_e, raw_i = raw_re, raw_ri
    else:
        raw = []
        for s in seeds:
            p = build_path(kind_prefix, wee_sc, wei_sc, s)
            raw.append(pd.read_csv(p).values[:-1])
        raw_e = raw
        raw_i = raw
    print(kind_prefix, np.shape(np.array(raw_e)))
    return np.stack(raw_e, axis=0), np.stack(raw_i, axis=0)

# ──────────────────────────────────────────────────────────────────────
#  Shared network and plasticity parameters
# ──────────────────────────────────────────────────────────────────────
lt = 1.0

N = 500
N_E = N
N_I = N
ms_per_sec = 1000
tau_rprim = 2.0 / ms_per_sec #irrelevant scaling factor soe weights can be written as integers
scale_factor = np.sqrt(N)
w_II = 10.0 / scale_factor * tau_rprim
w_IE = 10.0 / scale_factor * tau_rprim
aE = 20
tau_STDP = 120 / ms_per_sec
aI = aE
b = 20.0

c_x = 0.0
tau_wee = 225000.
tau_wei = 75000.
tau_r = 8.0 / ms_per_sec
tau_re = tau_r
tau_ri = tau_r * 2
tau_ou = 8.0 / ms_per_sec
w_IX = 15.0 / scale_factor * tau_rprim
w_EX = 20.0 / scale_factor * tau_rprim
Wx = np.array([[w_EX, 0], [0, w_IX]])

# ──────────────────────────────────────────────────────────────────────
#  Figure layout
# ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.5, 4.5), dpi=100)
grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1])

ax_plastic = fig.add_subplot(grid[0, 1])
ax1 = fig.add_subplot(grid[1, 0])
ax2 = fig.add_subplot(grid[0, 2])
ax4 = fig.add_subplot(grid[0, 3])
ax2_top = fig.add_subplot(grid[1, 1])
ax2_bottom = fig.add_subplot(grid[1, 2])
ax3 = fig.add_subplot(grid[1, 3])

# ──────────────────────────────────────────────────────────────────────
#  Plasticity-rule panel
# ──────────────────────────────────────────────────────────────────────
plot_plasticity_rule(ax_plastic, ft, lt, b)

# ──────────────────────────────────────────────────────────────────────
#  Fixed rates, feasible region, and stability boundaries in weight space
# ──────────────────────────────────────────────────────────────────────
rx = np.array([aE, aI])
max_wi = 40
w_EE = np.linspace(0.0, 35, 150) / scale_factor * tau_rprim
w_EI = np.linspace(0.0, max_wi, 150) / scale_factor * tau_rprim
w_ee, w_ei = np.meshgrid(w_EE, w_EI)
W = np.zeros((len(w_EE), len(w_EI), 2, 2))
W[..., 0, 0] = w_EE[:, None].T
W[..., 0, 1] = -w_EI[None, :].T
W[..., 1, 0] = w_IE
W[..., 1, 1] = -w_II

fixed_points = np.zeros(shape=(2, len(w_EE), len(w_EI)))
for i in range(len(w_EE)):
    for j in range(len(w_EI)):
        M = np.eye(2) - N * W[i, j, :, :]
        fixed_points[:, i, j] = np.linalg.inv(M) @ Wx @ rx * N

re_fixed = fixed_points[0, :, :]
ri_fixed = fixed_points[1, :, :]

dwee = re_fixed * (re_fixed - b) * tau_STDP / tau_wee
dwei = ri_fixed * (re_fixed - b) * tau_STDP / tau_wei

wEE_range = np.linspace(-1, w_EE[-1], 200)

num = (-b + b * N * wEE_range + aE * N * w_EX) * (1 + N * w_II)
den = (b * N**2 * w_IE + aI * N**2 * w_IX)
wEI_line = num / den
ax1.plot(wEE_range, wEI_line, color='b', linewidth=lt, label='wEI(wEE)')


axs = [ax2, ax1]
for ax in axs:
    # Draw the zero-rate boundaries.
    wEEval_rIzero = (aE * N * w_EX * w_IE + aI * w_IX) / (aI * N * w_IX)
    ax.axvline(wEEval_rIzero, color='k', linestyle=':', linewidth=lt)
    wEIval_rEzero = (aE * w_EX * (1 + N * w_II)) / (aI * N * w_IX)
    ax.axhline(wEIval_rEzero, color='k', linestyle=':', linewidth=lt)

    ax.set_ylim(0, w_EI[-1] + .0005)
    ax.set_xlim(0, w_EE[-1] + .0005)

    rax = ax.twinx()
    rax.set_ylim(ax.get_ylim())                                # ensure same scaling
    rax.set_yticks([wEIval_rEzero - 1e-4])
    rax.set_yticklabels([r'$r_E^*=0$'], fontsize=TICK_FS, rotation=45)

    osc = (tau_re + tau_ri + N * tau_re * w_II) / (N * tau_ri)
    print('osc,', osc)

    rax.spines['top'].set_visible(False)
    rax.spines['right'].set_visible(False)
    for label in rax.get_yticklabels():
        label.set_horizontalalignment('right')  # right align relative to position
        label.set_x(1.08)
        label.set_y(1.08)

    # Shade the feasible positive-rate region.
    cmap = ListedColormap([(1, 1, 1, 0), 'tab:green'])
    def condition(re_fixed):
        return re_fixed > 0
    def condition2(w_ei, wEIval_rEzero):
        return w_ei > (wEIval_rEzero)
    def condition3(w_ei, wEIval_rEzero):
        return w_ee < osc
    mask1 = condition(re_fixed)
    mask2 = condition2(w_ei, wEIval_rEzero)
    mask4 = condition2(w_ei, wEIval_rEzero)
    mask3 = np.logical_and(mask1, mask2)
    mask5 = np.logical_and(mask3, mask4)
    ax.pcolormesh(w_ee, w_ei, mask5, cmap=cmap, alpha=0.5)

    # Draw the non-oscillatory stability boundary.
    nonosc_stability_boundary = (-1 + N * w_EE - N * w_II + N**2 * w_EE * w_II) / (N**2 * w_IE)
    ax.set_title('Rates in Weight Space:', fontsize=TITLE_FS)
    ax.set_xlabel(r'$\bar{w}_{EE}$', fontsize=LABEL_FS, labelpad=5)
    ax.set_ylabel(r'$\bar{w}_{EI}$', fontsize=LABEL_FS, labelpad=5)

    idx = np.where(w_EE < wEEval_rIzero)
    ax.fill_between(w_EE[idx], nonosc_stability_boundary[idx], wEIval_rEzero, color='m', alpha=.25)
    ax.plot(w_EE, nonosc_stability_boundary, color='k', linestyle='--', linewidth=lt)
    # Format ticks and labels.
    ax.set_yticks([.001, .003], ['.001', '.003'], fontsize=TICK_FS, rotation=45)
    ax.text(wEEval_rIzero - 5e-4, -5e-4, r'$r_I^*=0$')
    ax.set_xticks([0, .001], ['0', '.001'], fontsize=TICK_FS)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    ax.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)


# ──────────────────────────────────────────────────────────────────────
#  Plasticity vector field
# ──────────────────────────────────────────────────────────────────────
w_EE = np.linspace(0, 35, 15) / scale_factor * tau_rprim
w_EI = np.linspace(0, max_wi, 15) / scale_factor * tau_rprim
w_ee, w_ei = np.meshgrid(w_EE, w_EI)
W = np.zeros((len(w_EE), len(w_EI), 2, 2))
W[..., 0, 0] = w_EE[:, None].T
W[..., 0, 1] = -w_EI[None, :].T
W[..., 1, 0] = w_IE
W[..., 1, 1] = -w_II

fixed_points = np.zeros(shape=(2, len(w_EE), len(w_EI)))
for i in range(len(w_EE)):
    for j in range(len(w_EI)):
        M = np.eye(2) - N * W[i, j, :, :]
        fixed_points[:, i, j] = np.linalg.inv(M) @ Wx @ rx * N

re_fixed = fixed_points[0, :, :]
ri_fixed = fixed_points[1, :, :]
mask1 = (re_fixed > 0)

dwee = re_fixed * (re_fixed - b) * tau_STDP / tau_wee
dwei = ri_fixed * (re_fixed - b) * tau_STDP / tau_wei
filtered_dwee = np.where(mask1, dwee, 0)
filtered_dwei = np.where(mask1, dwei, 0)
magnitude = np.sqrt(filtered_dwee**2 + filtered_dwei**2)
filtered_dwee = filtered_dwee / magnitude
filtered_dwei = filtered_dwei / magnitude
ax1.quiver(w_ee, w_ei, filtered_dwee, filtered_dwei, scale=12., width=0.005, scale_units='inches', color='k', alpha=.8, zorder=20)

# ──────────────────────────────────────────────────────────────────────
#  Stable and unstable initial-condition rate traces
# ──────────────────────────────────────────────────────────────────────
wie = w_IE
wii = w_II
w_EE_coords = np.array([18, 33]) / scale_factor * tau_rprim
cs = ['tab:green', 'tab:orange']
conditions = ['stable Sim', 'unstable Sim']

seeds = 1
T = 5.
ms_per_sec = 1000
sigma = 0.
dt = .1 / ms_per_sec
time = np.linspace(0, T, int(T / dt))
loc = 'RateDynamicsdata'
skip = 20
weival = 20. / scale_factor * tau_rprim

theory_label = ''
theory_colors = [dark_tab['th:green'], dark_tab['th:orange']]
for i in range(len(w_EE_coords)):
    weeval = w_EE_coords[i]
    color = cs[i]
    time_reduced = time[::skip]

    mean_theory_re_seeds = np.zeros(shape=(seeds, len(time)))
    mean_re_seeds = np.zeros(shape=(seeds, len(time_reduced)))
    mean_ri_seeds = np.zeros(shape=(seeds, len(time_reduced)))

    for seed in range(seeds):
        track_mean_rates = pd.read_csv(f'{loc}/mean_rates_Ne_{N_E}_T{T}_wee{weeval}_wei{weival}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv').to_numpy()
        mean_re_seeds[seed, :] = track_mean_rates[0, :-1]
        mean_ri_seeds[seed, :] = track_mean_rates[1, :-1]

        track_theory_rates = pd.read_csv(f'{loc}/re_theory_Ne_{N_E}_T{T}_wee{weeval}_wei{weival}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv').to_numpy()
        mean_theory_re_seeds[seed, :] = track_theory_rates[:, 0]

    idx = np.where(mean_re_seeds>100)
    mean_re_seeds[idx]=100

    ax2.scatter(weeval, weival, s=20, color=color)

    time_reduced = time_reduced[:-1]
    mean_re = np.mean(mean_re_seeds, axis=0)[:-1]
    std_re = np.std(mean_re_seeds, axis=0)[:-1]
    mean_ri = np.mean(mean_ri_seeds, axis=0)[:-1]
    std_ri = np.std(mean_ri_seeds, axis=0)[:-1]

    idx = np.where(time_reduced<110)
    line1, = ax4.plot(time_reduced[idx], mean_re[idx], color=color, label=conditions[i], linewidth=lt)
    ax4.fill_between(time_reduced[idx], mean_re[idx] - std_re[idx], mean_re[idx] + std_re[idx], alpha=.1, color=color)

    if i == 0:
        theory_label = 'Stable Theory'
    else:
        theory_label = 'Unstable Theory'
    idx = np.where(mean_theory_re_seeds>100)
    mean_theory_re_seeds[idx]=100
    mean_re = np.mean(mean_theory_re_seeds, axis=0)[:-1]
    line2, = ax4.plot(time[:-1], mean_re, color=theory_colors[i], linewidth=lt, label=theory_label)

ax4.set_ylim(0, 50)
ax4.set_xlim(0, .1)
ax4.set_xlabel('Time (ms)', fontsize=LABEL_FS, labelpad=-5)
ax4.set_title('Stable & Unstable I.C.s', fontsize=TITLE_FS)
ax4.set_xticks([0, .1], [0, 100], fontsize=TICK_FS)
ax4.set_yticks([0, 20, 40], [0, 20, 40], fontsize=TICK_FS)
ax4.set_ylabel(r'$\bar{r}_E$(t) (Hz)', fontsize=LABEL_FS)
ax4.legend(loc='upper right', frameon=False, fontsize=LEGEND_FS, bbox_to_anchor=(1.1, 1.05))

# ──────────────────────────────────────────────────────────────────────
#  Spiking simulation and theory: weights and rates over time
# ─────────────────────────────────────────────────────────────────────
seeds = 1
T = 110.
ms_per_sec = 1000
sigma = 0.
dt = .1 / ms_per_sec
time = np.linspace(0, T, int(T / dt))
print(np.shape(time))
loc = 'Fig3data'
w_EE = 16. / scale_factor * tau_rprim
wee = w_EE
w_EI = 25. / scale_factor * tau_rprim
wei = w_EI
skip = 40

label = ''
re_label = r'$\bar{r}_E$' + f'{label} '
ri_label = r'$\bar{r}_I$' + f'{label} '
wee_label = r'$\bar{w}_{EE}$' + f'{label} '
wei_label = r'$\bar{w}_{EI}$' + f'{label} '

time_reduced = time[::skip]
mean_wee_seeds = np.zeros(shape=(seeds, len(time_reduced)))
mean_wei_seeds = np.zeros(shape=(seeds, len(time_reduced)))
mean_re_seeds = np.zeros(shape=(seeds, len(time_reduced)))
mean_ri_seeds = np.zeros(shape=(seeds, len(time_reduced)))

theory_colors = [dark_tab['th:blue'], dark_tab['th:red']]
for seed in range(seeds):
    mean_wee = pd.read_csv(f'{loc}/mean_ee_weights_Ne_{N_E}_T{T}_wee{wee}_wei{wei}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv').to_numpy()[:-1, 0]
    mean_wei = pd.read_csv(f'{loc}/mean_ei_weights_Ne_{N_E}_T{T}_wee{wee}_wei{wei}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv').to_numpy()[:-1, 0]
    mean_wee_seeds[seed, :] = mean_wee
    mean_wei_seeds[seed, :] = mean_wei

    track_mean_rates = pd.read_csv(f'{loc}/mean_rates_Ne_{N_E}_T{T}_wee{wee}_wei{wei}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv').to_numpy()
    mean_re_seeds[seed, :] = track_mean_rates[0, :-1]
    mean_ri_seeds[seed, :] = track_mean_rates[1, :-1]

    df_Wee_theory = pd.read_csv(f'{loc}/W_theory_Ne_{N_E}_T{T}_wee{wee}_wei{wei}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv').to_numpy()

mean_wee = np.mean(mean_wee_seeds, axis=0)
std_wee = np.std(mean_wee_seeds, axis=0)
mean_wei = np.mean(np.abs(mean_wei_seeds), axis=0)
std_wei = np.std(np.abs(mean_wei_seeds), axis=0)

zord = 20
ax1.plot(mean_wee[:-1], mean_wei[:-1] / mean_wei[0], linewidth=lt, color='tab:red', linestyle='solid', zorder=zord)
ax2_top.fill_between(time_reduced, (mean_wee - std_wee) / mean_wee[0], (mean_wee + std_wee) / mean_wee[0], alpha=.3, color='tab:red', zorder=zord)
linewei, = ax2_top.plot(time_reduced, mean_wei / mean_wei[0], label=wei_label, color='b')
ax2_top.fill_between(time_reduced, (mean_wei - std_wei) / mean_wei[0], (mean_wei + std_wei) / mean_wei[0], alpha=.3, color='b', zorder=zord)
linewee, = ax2_top.plot(time_reduced, mean_wee / mean_wee[0], color='tab:red', label=wee_label, linewidth=lt)

time_reduced = time_reduced[:-1]
mean_re = np.mean(mean_re_seeds, axis=0)[:-1]
std_re = np.std(mean_re_seeds, axis=0)[:-1]
mean_ri = np.mean(mean_ri_seeds, axis=0)[:-1]
std_ri = np.std(mean_ri_seeds, axis=0)[:-1]

line1, = ax2_bottom.plot(time_reduced, mean_re, color='tab:red', label=re_label, linewidth=lt)
ax2_bottom.fill_between(time_reduced, mean_re - std_re, mean_re + std_re, alpha=.1, color='tab:red')
line2, = ax2_bottom.plot(time_reduced, mean_ri, color='tab:blue', label=ri_label, linewidth=lt)
ax2_bottom.fill_between(time_reduced, mean_ri - std_ri, mean_ri + std_ri, alpha=.1, color='tab:blue')

# Load and plot the population-theory overlays.
seeds = 1
dt = .1 / ms_per_sec
time = np.linspace(0, T, int(T / dt))

mean_theory_wee_seeds = np.zeros(shape=(seeds, len(time_reduced)))
mean_theory_wei_seeds = np.zeros(shape=(seeds, len(time_reduced)))
mean_theory_re_seeds = np.zeros(shape=(seeds, len(time_reduced)))
mean_theory_ri_seeds = np.zeros(shape=(seeds, len(time_reduced)))

for seed in range(seeds):
    mean_w = pd.read_csv(f'{loc}/W_theory_Ne_{N_E}_T{T}_wee{wee}_wei{wei}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv')
    print(np.shape(mean_w))
    mean_theory_wee_seeds[seed, :] = mean_w.to_numpy()[:-2, 0]

    mean_w = pd.read_csv(f'{loc}/Wei_theory_Ne_{N_E}_T{T}_wee{wee}_wei{wei}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv')
    mean_theory_wei_seeds[seed, :] = mean_w.to_numpy()[:-2, 0]

    track_mean_rates = pd.read_csv(f'{loc}/lamb_theory_Ne_{N_E}_T{T}_wee{wee}_wei{wei}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv').to_numpy()
    mean_theory_re_seeds[seed, :] = track_mean_rates[0, :-2]
    mean_theory_ri_seeds[seed, :] = track_mean_rates[1, :-2]


mean_theory_wee = np.mean(mean_theory_wee_seeds, axis=0)
mean_theory_wei = np.abs(np.mean(mean_theory_wei_seeds, axis=0))
mean_theory_re = np.mean(mean_theory_re_seeds, axis=0)
mean_theory_ri = np.mean(mean_theory_ri_seeds, axis=0)

ax1.plot(mean_theory_wee[:-5], mean_theory_wei[:-5], linewidth=lt, color='k', linestyle='solid')
ax1.scatter(mean_theory_wee[-5], mean_theory_wei[-5], s=2, color='k')
ax1.scatter(mean_theory_wee[0],  mean_theory_wei[0],  s=10, color='r')
linetheory, = ax2_top.plot(time_reduced, mean_theory_wee / mean_theory_wee[0], color=theory_colors[1], label=r'Theory $w_{EE}$', linewidth=lt, linestyle='solid')
linetheory2, = ax2_top.plot(time_reduced, mean_theory_wei / mean_theory_wei[0], color=theory_colors[0], label=r'Theory $w_{EI}$', linewidth=lt, linestyle='solid')

ax2_top.set_title('Weight Over Time', fontsize=TITLE_FS)
ax2_bottom.set_title('Rate Over Time', fontsize=TITLE_FS)

line3, = ax2_bottom.plot(time_reduced, mean_theory_re, linewidth=lt, color=theory_colors[1], linestyle='solid', label=r'Theory $r_E$')
line4, = ax2_bottom.plot(time_reduced, mean_theory_ri, linewidth=lt, color=theory_colors[0], linestyle='solid', label=r'Theory $r_I$')

ax2_top.set_ylim(0.25, 1.1)
ax2_top.set_yticks([0., 1])
ax2_top.set_yticklabels([0, 1], fontsize=TICK_FS)
ax2_top.set_xlim(0, 100)
ax2_top.set_xticks([0, 100])
ax2_top.set_xticklabels([0, 100], fontsize=TICK_FS)
ax2_top.set_xlabel("Time (sec)", fontsize=LABEL_FS)
ax2_top.set_ylabel(r"$\bar{w}/\bar{w}_{0}$", fontsize=LABEL_FS)

ax2_bottom.set_ylim(5, 25)
ax2_bottom.set_xlim(0, 100)
ax2_bottom.set_yticks([10, 20])
ax2_bottom.set_yticklabels([10, 20], fontsize=TICK_FS)
ax2_bottom.set_xticks([0, 100])
ax2_bottom.set_xticklabels([0, 100], fontsize=TICK_FS)
ax2_bottom.set_xlabel("Time (sec)", fontsize=LABEL_FS)
ax2_bottom.set_ylabel(r"$\bar{r}_E(t)$ (Hz)", fontsize=LABEL_FS)

# ──────────────────────────────────────────────────────────────────────
#  Attractors for different external excitatory input rates
# ──────────────────────────────────────────────────────────────────────
N = 500
N_E = N
N_I = N
ms_per_sec = 1000
tau_rprim = 2.0 / ms_per_sec
scale_factor = np.sqrt(N)
w_II = 10.0 / scale_factor * tau_rprim
w_IE = 10.0 / scale_factor * tau_rprim
aE = 20
aI = aE
b = 20.0
c_x = 0.0
tau_STDP = 120. / ms_per_sec
tau_wei = 75000.
tau_wee = 225000.
tau_r = 8.0 / ms_per_sec
tau_ou = 8.0 / ms_per_sec
w_IX = 15.0 / scale_factor * tau_rprim
w_EX = 20.0 / scale_factor * tau_rprim

aEs = np.array([10, 20, 30])
norm = colors.Normalize(vmin=min(aEs), vmax=max(aEs) + 10)
base_color = np.array([0.0, 1.0, 0.0])


def get_scaled_color(aE_val):
    scale = norm(aE_val)
    return base_color * scale


colors_scaled = [get_scaled_color(aE_val) for aE_val in aEs]

for i in range(len(aEs)):
    aE = aEs[i]
    color = colors_scaled[i]
    wEEval_rIzero = (aE * N * w_EX * w_IE + aI * w_IX) / (aI * N * w_IX)
    ax3.axvline(wEEval_rIzero, color=color, linestyle=':', linewidth=lt)
    wEIval_rEzero = (aE * w_EX * (1 + N * w_II)) / (aI * N * w_IX)
    ax3.axhline(wEIval_rEzero, color=color, linestyle=':', linewidth=lt)
    wEE_range = np.linspace(-1, wEEval_rIzero, 200)
    num = (-b + b * N * wEE_range + aE * N * w_EX) * (1 + N * w_II)
    den = (b * N**2 * w_IE + aI * N**2 * w_IX)
    wEI_line = num / den
    ax3.plot(wEE_range, wEI_line, color=color, linestyle='solid', linewidth=lt)
    ax3.text(0.32, 1.05 - 0.01 * aE, rf'$a_E$={aE}', color=color, fontsize=LABEL_FS, ha='right', va='top', transform=ax3.transAxes)
    ax3.set_xticks([], [])
    ax3.set_yticks([], [])

# Load final simulated weights and fit the displayed regression lines.
factor = np.sqrt(N_E)
sigma = 0
data_dir = Path("VaryaE")
seeds = [0]
T = 30.0
aEs = [10, 20, 30]

cfg = {
    10: ([20, 22, 24], [20]),
    20: ([16, 18, 20, 22, 24, 26], [33]),
    30: ([28, 30, 32], [55]),
}

for aE in aEs:
    wEEs, wEIs = cfg[aE]
    final_wEE = []
    final_wEI = []
    for wee_raw in wEEs:
        for wei_raw in wEIs:
            _re, _ri = load_csv_stack(f"aE{aE}_mean_rates_", wee_raw, wei_raw)
            w_ee_arr = load_csv_stack(f"aE{aE}_mean_ee_weights_", wee_raw, wei_raw)[0].squeeze()
            w_ei_arr = load_csv_stack(f"aE{aE}_mean_ei_weights_", wee_raw, wei_raw)[0].squeeze()
            final_wEE.append(w_ee_arr[-1])
            final_wEI.append(np.abs(w_ei_arr[-1]))

    X = np.array(final_wEE).reshape(-1, 1)
    y = np.array(final_wEI)
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    wEEval_rIzero = (aE * N * w_EX * w_IE + aI * w_IX) / (aI * N * w_IX)
    wEE_range = np.linspace(-1.0, wEEval_rIzero, 200)
    y_pred = model.predict(wEE_range.reshape(-1, 1))
    order = np.argsort(wEE_range)
    ax3.plot(wEE_range[order], y_pred[order], linestyle="--", color="r", linewidth=lt)
    print(f"R² value (aE={aE}): {r2:.4f}")

ax3.set_title(r"Attractor for different $a_E$", fontsize=TITLE_FS)
ax3.set_xlabel(r"$\bar{w}_{EE}$", fontsize=LABEL_FS)
ax3.set_ylabel(r"$\bar{w}_{EI}$", fontsize=LABEL_FS)

ytpos = [0.002, 0.006]
ytlabels = [".002", ".006"]
ax3.set_yticks(ytpos)
ax3.set_yticklabels(ytlabels, fontsize=TICK_FS, rotation=45)

xtpos = [0, 0.001, 0.004]
xtlabels = ['0', '.001', '.004']
ax3.set_xticks(xtpos)
ax3.set_xticklabels(xtlabels, fontsize=TICK_FS)
ax3.set_ylim(0, 0.006)
ax3.set_xlim(0, 0.004)
ax3.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
ax3.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax3.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)

ax3.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
ax3.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)

ax3.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
ax1.set_title('Flow in Weight Space', fontsize=TITLE_FS)

plt.subplots_adjust(left=.04, bottom=.08, wspace=.35, top=.91, hspace=.4, right=.98)

# ──────────────────────────────────────────────────────────────────────
#  Final cosmetic pass (spines, ticks, titles, legends)
# ──────────────────────────────────────────────────────────────────────
axs_all = [ax_plastic, ax1, ax2_top, ax2_bottom, ax3, ax2, ax4]
for ax in axs_all:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(lt)
    ax.spines['left'].set_linewidth(lt)
    ax.tick_params(axis='both', labelsize=TICK_FS)
    ax.set_title(ax.get_title(), fontsize=TITLE_FS)

for a in [ax4, ax2_top, ax2_bottom, ax_plastic]:
    a.xaxis.labelpad = -8
    a.yaxis.labelpad = 0

# Add the final legends.
ax2_bottom.legend(handles=[line1, line2, line3,line4], bbox_to_anchor=(1.1, 0.05), loc="lower right", fontsize=LEGEND_FS, frameon=False, ncol=2)
ax2_top.legend(handles=[linewee, linewei, linetheory,linetheory2], bbox_to_anchor=(1.16, .35), loc="upper right", fontsize=LEGEND_FS, frameon=False, ncol=2)

for a in [ax1, ax2, ax3]:
    a.xaxis.labelpad = -10
    a.yaxis.labelpad = -12


plt.rcParams.update({
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
})

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}"})

plt.savefig('Fig3.pdf', dpi=200)
plt.show()

