from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Figure-wide font settings.
TITLE_FS = 11
LABEL_FS = 11
TICK_FS = 8
LEGEND_FS = 9

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "text.latex.preamble": r"\usepackage{amsmath}"})


# Data location and simulation parameters encoded in the CSV filenames.
data_dir = Path('Fig2data')
T = 40.0
N_E = 500
factor = np.sqrt(N_E)
tau_rprim = 2.0 / 1000.0
w_IE = 10. / factor * tau_rprim
w_II = 10. / factor * tau_rprim

mc_per_sec = 1000
sigma = 0
c_x = 0.
tau_ou = 8. / mc_per_sec
tau_r = 8. / mc_per_sec
tau_STDP = 120. / mc_per_sec
tau_wei = 75000.
tau_wee = 225000.

r_EX = 20
wX = tau_rprim * np.array([20., 15]) / np.sqrt(N_E)


def scaled_value(raw: float) -> float:
    """Scale a raw population weight into the units used by the simulation."""
    return raw / factor * tau_rprim


def build_path(kind: str, wee_sc: float, wei_sc: float, seed: int) -> Path:
    """Build the path to one simulation CSV."""
    fname = f'{kind}Ne_{N_E}_T{T}_wee{wee_sc}_wei{wei_sc}_wii{w_II*factor}_wie{w_IE*factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
    return data_dir / fname


def load_csv_stack(kind_prefix: str, wee_raw: float, wei_raw: float):
    """Load rate or weight CSVs across the seeds selected below."""
    wee_sc = np.around(scaled_value(wee_raw) * factor, 3)
    wei_sc = np.around(scaled_value(wei_raw) * factor, 3)

    if 'rates' in kind_prefix:
        raw_re, raw_ri = [], []
        for s in seeds:
            p = build_path(kind_prefix, wee_sc, wei_sc, s)
            rates = pd.read_csv(p).values
            raw_re.append(rates[0, :-1])
            raw_ri.append(rates[1, :-1])
        return np.stack(raw_re, axis=0), np.stack(raw_ri, axis=0)

    raw = []
    for s in seeds:
        p = build_path(kind_prefix, wee_sc, wei_sc, s)
        raw.append(pd.read_csv(p).values[:-1])
    return raw


# Create the six Figure 2 plotting regions.
lt = 1.5
fig = plt.figure(figsize=(7.5, 4.5), dpi=100)
grid = fig.add_gridspec(2, 3)

ax_init_grid = fig.add_subplot(grid[0, 2])
ax_init_weights = fig.add_subplot(grid[0, 0])
ax_init_rates = fig.add_subplot(grid[1, 0])
ax_init_wei = fig.add_subplot(grid[0, 1])
ax_lineattr_rate_var = fig.add_subplot(grid[1, 2])

try:
    _sub = grid[1, 1].subgridspec(2, 1, hspace=0.30)
    ax_lineattr_rate_ri = fig.add_subplot(_sub[0, 0])
    ax_lineattr_rate = fig.add_subplot(_sub[1, 0])
except Exception:
    # Existing fallback behavior is retained; see README for the associated compatibility note.
    pass


# Configure the weight-space panel before loading the trajectories.
ax_init_grid.set_xlabel(r'$\bar{w}_{EE}$', fontsize=LABEL_FS)
ax_init_grid.set_ylabel(r'$\bar{w}_{EI}$', fontsize=LABEL_FS, labelpad=-5)
ax_init_grid.set_title('Different Initial conditions', fontsize=TITLE_FS)
ax_init_grid.set_xlim(0, scaled_value(20) * 1.1)
ax_init_grid.set_ylim(0, scaled_value(30) * 1.1)
ax_init_grid.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_init_grid.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_init_grid.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
ax_init_grid.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)


# Initial conditions and storage for the line-attractor analyses.
w_EEs = np.array([25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14])
w_EIs = np.array([30])
norm = colors.Normalize(vmin=8, vmax=29)

final_wEE_list, final_wEI_list = [], []
mean_rate_window, var_rate_window = [], []
colour_store = []
mean_rate_ri_window = []
error_var = []
mean_rate_err_window = []
mean_rate_ri_err_window = []
labels = ['High', 'Low']
ct = 0


# Load each initial condition, plot its trajectory, and collect late-time statistics.
print('========== LOADING MULTI-ICs =========')
for wee_raw in w_EEs:
    for wei_raw in w_EIs:
        seeds = np.arange(10)

        re, ri = load_csv_stack('mean_rates_', wee_raw, wei_raw)
        w_ee = load_csv_stack('mean_ee_weights_', wee_raw, wei_raw)
        w_ei = load_csv_stack('mean_ei_weights_', wee_raw, wei_raw)

        w_ee = np.mean(w_ee, axis=0).squeeze()
        w_ei = np.mean(w_ei, axis=0).squeeze()
        print('np.shape()', re.shape)
        print('w_ee shape', np.shape(w_ee))

        L = len(w_ee)
        time = np.linspace(0, T, L)
        maxval = T
        mask = (time >= 0) & (time <= maxval)

        color = cm.plasma(norm(wee_raw))
        colour_store.append(color)

        ax_init_grid.plot(w_ee[mask], np.abs(w_ei[mask]), color=color)
        ax_init_grid.scatter(w_ee[mask][-1], np.abs(w_ei[mask][-1]), color='k', s=20, zorder=30)

        if wei_raw == w_EIs[0]:
            ax_init_weights.plot(time[mask], w_ee[mask], color=color)
            ax_init_wei.plot(time[mask], np.abs(w_ei[mask]), color=color)

        if wee_raw == 14 or wee_raw == 25:
            ax_init_rates.plot(time[:-1], np.mean(re[:, :-1], axis=0), linewidth=.5, color=color, zorder=50 - wee_raw, label=labels[ct])
            ax_init_rates.fill_between(time[:-1], np.mean(re[:, :-1], axis=0) - np.std(re[:, :-1], axis=0), np.mean(re[:, :-1], axis=0) + np.std(re[:, :-1], axis=0), alpha=.5, color=color, zorder=50 - wee_raw)
            ct += 1

        final_wEE_list.append(w_ee[mask][-1])
        final_wEI_list.append(np.abs(w_ei[mask][-1]))

        # Measure firing-rate statistics over the retained 25-40 s analysis window.
        idx_start = np.searchsorted(time, 25.0)
        idx_end = np.searchsorted(time, maxval)

        mean_ri_window = np.mean(ri[:, idx_start:idx_end], axis=0)
        mean_rate_ri_window.append(np.mean(mean_ri_window))
        mean_rate_ri_err_window.append(np.std(mean_ri_window))

        mean_r_window = np.mean(re[:, idx_start:idx_end], axis=0)
        mean_rate_window.append(np.mean(mean_r_window))
        mean_rate_err_window.append(np.std(mean_r_window))

        var_r_window = np.var(re[:, idx_start:idx_end], axis=1)
        var_rate_window.append(np.mean(var_r_window, axis=0))
        error_var.append(np.std(var_r_window, axis=0) / np.sqrt(len(re[:, 0])))

        if wee_raw == w_EEs[-1]:
            print('NE', N_E)
            bigW = np.array([[(N_E - 1) * w_ee[mask][-1], -N_E * w_ei[mask][-1]], [N_E * w_IE, -(N_E - 1) * w_II]])
            print('(r_EX*N_E*wX)', (r_EX * N_E * wX))
            steady_state = np.linalg.inv(np.eye(2) - bigW) @ (r_EX * N_E * wX)


# Format the trajectory and example-rate panels.
print(mean_rate_err_window)
mean_rate_err_window = np.array(mean_rate_err_window) / np.sqrt(len(seeds))
mean_rate_ri_err_window = np.array(mean_rate_ri_err_window) / np.sqrt(len(seeds))

ax_init_weights.set_yticks([1.e-3, 2.e-3], ['.001', '.002'])
ax_init_weights.set_xticks([0., 40], [0., 40], fontsize=TICK_FS)
ax_init_weights.set_xlim(0, T)
ax_init_weights.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_init_weights.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)

ax_init_rates.set_xticks([0., 40], [0, 40])
ax_init_rates.set_yticks([0., 30], [0, 30])
ax_init_weights.set_title(r'$\bar{w}_{EE}$ over time', fontsize=TITLE_FS)
ax_init_wei.set_xticks([0., 30], [0, 30], fontsize=TICK_FS)
ax_init_wei.set_yticks([1.e-3, 2.e-3])
ax_init_wei.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_init_wei.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
ax_init_wei.set_xlim(0, T)
ax_init_wei.set_title(r'$\bar{w}_{EI}$ over time', fontsize=TITLE_FS)
ax_init_rates.legend(loc=(.6, .15), frameon=False, fontsize=LEGEND_FS)
ax_init_rates.set_title(r'$\bar{r}_E$ for Two Trajectories', fontsize=TITLE_FS)
print('steady_state', steady_state)


# Fit the final weight coordinates and draw the empirical line attractor.
X = np.array(final_wEE_list).reshape(-1, 1)
y = np.array(final_wEI_list)
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)
print(f"Line-attractor regression R² = {r_squared:.4f}")

X_wee = np.array([scaled_value(np.arange(30))]).reshape(-1, 1)
y_pred = model.predict(X_wee)
print(np.shape(y_pred))

sorted_indices = np.argsort(X_wee[:, 0])
ax_init_grid.plot(X_wee[sorted_indices, :], y_pred[sorted_indices], 'r--', label='Line attractor')
ax_init_grid.legend(loc='lower right', fontsize=LEGEND_FS, frameon=False)


# Project the final points onto the fitted line to define a normalized line coordinate.
final_wEE_list = np.array(final_wEE_list) / tau_rprim * factor
final_wEI_list = np.array(final_wEI_list) / tau_rprim * factor
np.savetxt('final_wEE_list.txt', final_wEE_list)
np.savetxt('final_wEI_list.txt', final_wEI_list)

model = LinearRegression().fit(final_wEE_list.reshape(-1, 1), final_wEI_list)
points = np.column_stack((final_wEE_list, final_wEI_list))
slope = model.coef_[0]
print('model.coef_[0]', slope)
direction = np.array([1.0, slope])
direction /= np.linalg.norm(direction)
p0 = np.array([0.0, model.intercept_])
s_coord = []
for pt in points:
    pt_vec = pt - p0
    s_coord.append(np.dot(pt_vec, direction))
print('s_coord', s_coord)
s_coord = np.array(s_coord)
s_coord = (s_coord - s_coord.min()) / (s_coord.max() - s_coord.min())


# Plot late-time E/I firing rates and E-rate variance along the line attractor.
for s, v, r, ri, c, err_v, err_re, err_ri in zip(s_coord, var_rate_window, mean_rate_window, mean_rate_ri_window, colour_store, error_var, mean_rate_err_window, mean_rate_ri_err_window):
    ax_lineattr_rate.scatter(s, r, color=c, s=20)
    ax_lineattr_rate.errorbar(s, r, yerr=err_re, color=c, markersize=8)
    ax_lineattr_rate_ri.scatter(s, ri, color=c, s=20)
    ax_lineattr_rate_ri.errorbar(s, ri, yerr=err_ri, color=c, markersize=8)
    ax_lineattr_rate_var.scatter(s, v, color=c, s=20)
    ax_lineattr_rate_var.errorbar(s, v, yerr=err_v, color=c, markersize=8)

ax_lineattr_rate.axhline(20, color='k', linestyle=':')
ax_lineattr_rate.set_xlabel('Location on Line', fontsize=LABEL_FS)
ax_lineattr_rate.set_ylabel(r'$\bar{r}_E$', fontsize=LABEL_FS)
ax_lineattr_rate.set_ylim(0, 25)
ax_lineattr_rate.set_xticks([0, 1.], [0, 1], fontsize=TICK_FS)
ax_lineattr_rate.set_xlim(0, 1)
ax_lineattr_rate.set_yticks([0, 20.], [0, 20], fontsize=TICK_FS)

ax_lineattr_rate_var.set_ylim(0, 41)
ax_lineattr_rate_var.set_ylabel(r'Var $\bar{r}_E$', fontsize=LABEL_FS)
ax_lineattr_rate_var.set_xlabel('Location on Line', fontsize=LABEL_FS)
ax_lineattr_rate_var.set_title(r'Var $\bar{r}_E$ vs line-pos.', fontsize=TITLE_FS)
ax_lineattr_rate_var.set_xticks([0, 1], [0, 1], fontsize=TICK_FS)
ax_lineattr_rate_var.set_yticks([0, 40], [0, 40], fontsize=TICK_FS)

ax_lineattr_rate_ri.set_ylim(0, 25)
ax_lineattr_rate_ri.set_xticks([0, 1], [0, 1], fontsize=TICK_FS)
ax_lineattr_rate_ri.set_xlim(0, 1)
ax_lineattr_rate_ri.set_ylabel(r'$\bar{r}_I$', fontsize=LABEL_FS)
ax_lineattr_rate_ri.set_title(r'Steady-state $\bar{r}$ vs line-pos.', fontsize=TITLE_FS)
ax_lineattr_rate_ri.set_yticks([0, 20.], [0, 20], fontsize=TICK_FS)


# Apply the remaining labels, ticks, and shared axis styling.
ax_init_grid.set_xticks([0, .003], [0, '.003'], fontsize=TICK_FS)
ax_init_grid.set_yticks([0, .003], [0, '.003'], fontsize=TICK_FS)
ax_init_grid.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_init_grid.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_init_grid.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
ax_init_grid.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
ax_init_wei.set_xlabel('Time (sec)', fontsize=LABEL_FS, labelpad=-5)
ax_init_weights.set_xlabel('Time (sec)', fontsize=LABEL_FS, labelpad=-10)
ax_init_weights.set_ylabel(r'$\bar{w}_{EE}$', fontsize=LABEL_FS)
ax_init_wei.set_ylabel(r'$\bar{w}_{EI}$', fontsize=LABEL_FS)
ax_init_rates.set_xlabel('Time (sec)', fontsize=LABEL_FS)
ax_init_rates.set_ylabel(r'$\bar{r}_E$', fontsize=LABEL_FS)

for ax in (ax_lineattr_rate_ri, ax_init_wei, ax_init_grid, ax_init_weights, ax_init_rates, ax_lineattr_rate, ax_lineattr_rate_var):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(lt)
    ax.spines['bottom'].set_linewidth(lt)
    ax.tick_params(labelsize=TICK_FS)

for a in (ax_lineattr_rate_ri, ax_init_grid, ax_init_weights, ax_init_rates, ax_lineattr_rate, ax_lineattr_rate_var):
    a.xaxis.labelpad = -5
    a.yaxis.labelpad = -5

for a in (ax_init_grid, ax_init_weights, ax_init_wei):
    a.yaxis.labelpad = -5

plt.rcParams.update({"axes.titlesize": TITLE_FS, "axes.labelsize": LABEL_FS, "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS, "legend.fontsize": LEGEND_FS})
plt.subplots_adjust(left=.05, bottom=.08, wspace=.35, top=.9, hspace=.35, right=.98)

# plt.savefig('Fig2.pdf', dpi=150)
plt.show()
