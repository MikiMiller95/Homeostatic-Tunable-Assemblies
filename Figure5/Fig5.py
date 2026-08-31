"""Create Figure 5 from deterministic theory and precomputed spiking simulations."""

import re as _re
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from numpy.linalg import eig
from sklearn.linear_model import LinearRegression

from Fig5Sim import run_sim
from helper_functions import calculate_CSD

# ──────────────────────────────────────────────────────────────────────────
# Global plotting style
# ──────────────────────────────────────────────────────────────────────────
TITLE_FS = 9
LABEL_FS = 9
TICK_FS = 7
LEGEND_FS = 8
plt.rcParams.update({'axes.titlesize': TITLE_FS, 'axes.labelsize': LABEL_FS, 'xtick.labelsize': TICK_FS, 'ytick.labelsize': TICK_FS, 'legend.fontsize': LEGEND_FS})
mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'text.latex.preamble': '\\usepackage{amsmath}'})
dark_tab = {'th:blue': '#0e3651', 'th:orange': '#733906', 'th:green': '#144814', 'th:red': '#601212', 'th:purple': '#432e55', 'th:brown': '#3f2722', 'th:pink': '#663657', 'th:gray': '#393939', 'th:olive': '#55550f', 'th:cyan': '#0a565d'}

# ──────────────────────────────────────────────────────────────────────────
# CSV filename and loading utilities
# ──────────────────────────────────────────────────────────────────────────
def fmt_trunc(x, places=18):
    d = Decimal(str(x)).quantize(Decimal(f'1e-{places}'), rounding=ROUND_DOWN)
    return format(d, 'f')
plasma = plt.get_cmap('plasma')
colors = [plasma(i) for i in np.linspace(0, 1, 22)]
_RX_WEE_WEI = _re.compile('_wee([-+0-9.eE]+)_wei([-+0-9.eE]+)_')

def build_path(kind, wee_sc, wei_sc, seed, Tlen, tol=1e-12):
    if 'rates' in kind:
        fname = f'{kind}Ne_{N}_T{float(Tlen)}_wee{_tok(wee_sc)}_wei{_tok(wei_sc)}_wii{_tok(w_II)}_wie{_tok(w_IE)}_sigma{_tok(sigma)}_cx{float(c_x)}_taur{_tok(tau_r)}_tauSTDP{_tok(tau_STDP)}_tauou{_tok(tau_ou)}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
    else:
        fname = f'{kind}_{N}_T{float(Tlen)}_wee{_tok(wee_sc)}_wei{_tok(wei_sc)}_wii{_tok(w_II)}_wie{_tok(w_IE)}_sigma{_tok(sigma)}_cx{float(c_x)}_taur{_tok(tau_r)}_tauSTDP{_tok(tau_STDP)}_tauou{_tok(tau_ou)}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
    p = data_dir / fname
    if p.exists():
        return p
    if 'rates' in kind:
        pat = data_dir / f'{kind}Ne_{N}_T{Tlen}_wee*_wei*_*_seed{seed}.csv'
    else:
        pat = data_dir / f'{kind}_{N}_T{Tlen}_wee*_wei*_*_seed{seed}.csv'
    cands = [q for q in pat.parent.glob(pat.name) if _RX_WEE_WEI.search(q.name)]
    if not cands:
        return p
    tgt_wee = float(wee_sc)
    tgt_wei = float(wei_sc)

    def dist(q):
        a, b = map(float, _RX_WEE_WEI.search(q.name).groups())
        return abs(a - tgt_wee) + abs(b - tgt_wei)
    best = min(cands, key=dist)
    d = dist(best)
    if d <= tol:
        return best
    return p

def _tok(x, places=19):
    d = Decimal(str(float(x))).quantize(Decimal(f'1e-{places}'), rounding=ROUND_HALF_UP)
    s = format(d, 'f').rstrip('0').rstrip('.')
    return s or '0'

def load_stack(kind_prefix, wee_raw, wei_raw, Tlen, seeds, mode='rates', weighted=False):
    if weighted:
        wee_sc = np.around(sc(wee_raw) * scale_factor, 3)
        wei_sc = np.around(sc(wei_raw) * scale_factor, 3)
    else:
        wee_sc, wei_sc = (sc(wee_raw), sc(wei_raw))
    stacks = []
    if wee_raw == 9.0:
        wei_sc = fmt_trunc(wei_sc)
    for s in seeds:
        f = build_path(kind_prefix, wee_sc, wei_sc, s, Tlen)
        arr = pd.read_csv(f).values
        if mode == 'weights':
            stacks.append(arr[:, 0])
        elif mode == 'rates':
            stacks.append(arr[:, :-1])
    return np.stack(stacks, axis=0)

def s_coord_fun(wEE_range, N):
    print('wEE_range', wEE_range)
    num = (-b + b * N * wEE_range + aE * N * w_EX) * (1 + N * w_II)
    den = b * N ** 2 * w_IE + aI * N ** 2 * w_IX
    wEI_line = num / den
    points = np.column_stack((wEE_range, wEI_line))
    model = LinearRegression().fit(wEE_range.reshape(-1, 1), wEI_line)
    slope = model.coef_[0]
    direction = np.array([1.0, slope])
    direction /= np.linalg.norm(direction)
    p0 = np.array([0.0, model.intercept_])
    s_coord = []
    for pt in points:
        pt_vec = pt - p0
        s_coord.append(np.dot(pt_vec, direction))
    s_coord = np.array(s_coord)
    return (s_coord, points, p0, direction)

# ──────────────────────────────────────────────────────────────────────────
# Autocovariance and stationary-covariance helpers
# ──────────────────────────────────────────────────────────────────────────
def acf_from_mask(re, mask2, dt, max_lag_s=1.0, unbiased=False, normalize=False):
    x = np.asarray(re, float)
    if x.ndim == 1:
        x = x[None, :]
    x = x[:, mask2]
    x = x - x.mean(axis=1, keepdims=True)
    n_series, T = x.shape
    L = min(T - 1, int(np.floor(max_lag_s / dt)))
    acov = np.empty((n_series, 2 * L + 1))
    lags = np.arange(-L, L + 1)
    for i in range(n_series):
        xi = x[i]
        ac0 = []
        for k in range(0, L + 1):
            v = np.dot(xi[:T - k], xi[k:])
            denom = T - k if unbiased else T
            ac0.append(v / denom)
        ac0 = np.array(ac0)
        acov[i, L:] = ac0
        acov[i, :L] = ac0[1:][::-1]
    if normalize:
        acov = acov / acov[:, [L]]
    return (lags * dt, acov)

def sigma_rr(W, Wx, tau_r, N, ext='poisson', rec='poisson', nu_ext=None, tau_ext=None, sigma_ext=None, tau_ou_ext=None, A_ext=1.0, nu_rec=None, tau_rec=None, sigma_rec=None, tau_ou_rec=None, A_rec=1.0, poisson_white_limit=False, r_x=None):
    """
    Stationary Σ_rr (2x2) for ṙ = K r + R_ext s_ext + R_rec s_rec.
    ext/rec in {'poisson','ou','none'}. For Poisson with exp kernel use tau_s_*.
    Set poisson_white_limit=True to use the classic white-shot-noise Lyapunov.
    Returns (Sigma_rr, Sigma_rr[0,0]).
    """
    W = np.asarray(W, float)
    Wx = np.asarray(Wx, float)
    tau = np.asarray(tau_r, float)
    D = np.diag(1.0 / tau)
    K = D @ (-np.eye(2) + N * W)
    R_ext = N * (D @ Wx)
    R_rec = N * (D @ W)
    Qwhite = np.zeros((2, 2), float)
    colored = []
    if ext == 'poisson':
        if poisson_white_limit:
            Qwhite += R_ext @ np.diag(A_ext ** 2 * nu_ext) @ R_ext.T
        else:
            alpha = 1.0 / tau_ext
            var = A_ext ** 2 * nu_ext / (2.0 * tau_ext)
            colored.append((R_ext, alpha, var))
    elif ext == 'ou':
        alpha = 1.0 / tau_ou_ext
        var = np.array([sigma_ext, sigma_ext]) ** 2
        colored.append((R_ext, alpha, var))
    if rec == 'poisson':
        if poisson_white_limit:
            Qwhite += R_rec @ np.diag(A_rec ** 2 * nu_rec) @ R_rec.T
        else:
            alpha = 1.0 / tau_rec
            var = A_rec ** 2 * nu_rec / (2.0 * tau_ext)
            colored.append((R_rec, alpha, var))
    elif rec == 'ou':
        alpha = 1.0 / tau_ou_rec
        var = np.array([sigma_rec, sigma_rec]) ** 2
        colored.append((R_rec, alpha, var))
    m = 2 + 2 * len(colored)
    A = np.zeros((m, m), float)
    Q = np.zeros((m, m), float)
    A[:2, :2] = K
    Q[:2, :2] = Qwhite
    j = 2
    for R, alpha, var in colored:
        A[:2, j:j + 2] = R
        A[j:j + 2, j:j + 2] = -np.diag(alpha)
        Q[j:j + 2, j:j + 2] = np.diag(2.0 * alpha * var)
        j += 2
    n = m
    L = np.kron(np.eye(n), A) + np.kron(A, np.eye(n))
    s = np.linalg.solve(L, -Q.reshape(n * n, order='F'))
    Sigma = s.reshape(n, n, order='F')
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma_rr = Sigma[:2, :2].real
    return (Sigma_rr / N, float(Sigma_rr[0, 0]) / N)
# ──────────────────────────────────────────────────────────────────────────
# Shared network and plasticity parameters
# ──────────────────────────────────────────────────────────────────────────
N = 500
ms_per_sec = 1000
tau_rprim = 2.0 / ms_per_sec
scale_factor = np.sqrt(N)

def sc(raw):
    """Helper: raw weight (Hz) → simulation units used everywhere."""
    return raw / scale_factor * tau_rprim

def rev_sc(raw):
    """Helper: raw weight (Hz) → simulation units used everywhere."""
    return raw * scale_factor / tau_rprim
WEIGHTS_RAW = dict(IC1=dict(w_EE=28.0, w_EI=None), IC2=dict(w_EE=10.0, w_EI=None))
w_IE = sc(10.0)
w_II = sc(10.0)
w_EX = sc(20.0)
w_IX = sc(15.0)
W_x = np.array([[w_EX, 0], [0, w_IX]])
tau_STDP = 120.0 / ms_per_sec
tau_wee = 225000.0
tau_wei = 75000.0
tau_r = 8.0 / ms_per_sec
tau_ou = 8.0 / ms_per_sec
factor = np.sqrt(N)
aE = aI = 20.0
b = 20.0
c_x = 0.0
SIMLEN_IC = 7.5
# ──────────────────────────────────────────────────────────────────────────
# Figure layout
# ──────────────────────────────────────────────────────────────────────────
lt, ft = (1.0, 7)
fig = plt.figure(figsize=(7.5, 4.5), dpi=200)
grid = fig.add_gridspec(2, 5, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1])
ax2 = fig.add_subplot(grid[0, 0])
ax3_IC1 = fig.add_subplot(grid[0, 2])
ax3_IC2 = fig.add_subplot(grid[0, 1])
ax_tau = fig.add_subplot(grid[1, 1])
ax_energy = fig.add_subplot(grid[1, 0])
ax_var = fig.add_subplot(grid[1, 2])
ax_SNR = fig.add_subplot(grid[1, 3])
ax_balance = fig.add_subplot(grid[1, 4])
ax_corr_th = fig.add_subplot(grid[0, 4])
ax_corr = fig.add_subplot(grid[0, 3])
Wx = np.array([[w_EX, 0], [0, w_IX]])
rx = np.array([aE, aI])
# ──────────────────────────────────────────────────────────────────────────
# Weight-space geometry and admissible region
# ──────────────────────────────────────────────────────────────────────────
print('plotting the heat map')
w_EE_vals = np.linspace(0.0, 37.0, 400) / scale_factor * tau_rprim
w_EI_vals = np.linspace(0.0, 45.0, 400) / scale_factor * tau_rprim
w_ee, w_ei = np.meshgrid(w_EE_vals, w_EI_vals)
W = np.zeros((len(w_EE_vals), len(w_EI_vals), 2, 2))
W[..., 0, 0] = w_EE_vals[:, None].T
W[..., 0, 1] = -w_EI_vals[None, :].T
W[..., 1, 0] = w_IE
W[..., 1, 1] = -w_II
fixed = np.zeros((2, len(w_EE_vals), len(w_EI_vals)))
for i in range(len(w_EE_vals)):
    for j in range(len(w_EI_vals)):
        M = np.eye(2) - N * W[i, j]
        fixed[:, i, j] = np.linalg.inv(M) @ (Wx @ rx * N)
re_fixed, ri_fixed = fixed
mask_pos = (re_fixed > 0) & (ri_fixed > 0)
cmap_mask = ListedColormap([(1, 1, 1, 0), 'm'])
ax2.pcolormesh(w_ee, w_ei, mask_pos, cmap=cmap_mask, alpha=0.4)
axs = [ax2]
for ax in axs:
    wEEval_rIzero = (aE * N * w_EX * w_IE + aI * w_IX) / (aI * N * w_IX)
    ax.axvline(wEEval_rIzero, color='k', linestyle=':', linewidth=lt)
    wEIval_rEzero = aE * w_EX * (1 + N * w_II) / (aI * N * w_IX)
    ax.axhline(wEIval_rEzero, color='k', linestyle=':', linewidth=lt)
    wEE_range = np.linspace(0, w_EE_vals[-1], 100)
    num = (-b + b * N * wEE_range + aE * N * w_EX) * (1 + N * w_II)
    den = b * N ** 2 * w_IE + aI * N ** 2 * w_IX
    wEI_line = num / den
    ax.plot(wEE_range, wEI_line, color='b', linewidth=lt)
    ax.set_ylim(0, w_EI_vals[-1])
    ax.set_xlim(0, w_EE_vals[-1] + 0.0001)
    pos = [0.001, 0.003]
    labels = ['.001', '.003']
    ax.set_yticks(pos, labels)
    ax.text(0.95, wEIval_rEzero, '$r_E^*=0$')
    ax.set_yticklabels(['0', '0.001'])
    pos = [0, 0.001]
    labels = ['0', '.001']
    ax.text(wEEval_rIzero, -0.0002, '$r_I^*=0$')
    ax.set_xticks(pos, labels)
    yt = ax.get_yticklabels()
    label_re_zero = yt[-1]
    label_re_zero.set_horizontalalignment('right')
    label_re_zero.set_x(1.3)
    label_re_zero.set_y(1.08)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3), useMathText=True)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, -3), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    ax.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
# ──────────────────────────────────────────────────────────────────────────
# Deterministic pulse responses at two line-attractor locations
# ──────────────────────────────────────────────────────────────────────────
print('running the perturb sim')
idxcol = 3
max_time = 7.49
# Run deterministic responses for the high and low initial conditions.
for key, clr, color_trace in zip(('IC1', 'IC2'), ('tab:orange', colors[idxcol]), (dark_tab['th:orange'], dark_tab['th:purple'])):
    w_EE0 = sc(WEIGHTS_RAW[key]['w_EE'])
    num = (-b + b * N * w_EE0 + aE * N * w_EX) * (1 + N * w_II)
    den = b * N ** 2 * w_IE + aI * N ** 2 * w_IX
    w_EI0 = num / den
    if key == 'IC2':
        ax_plot = ax3_IC2
        ax_plot.set_title('Low on Line')
    else:
        ax_plot = ax3_IC1
        ax_plot.set_title('High on Line')
    ax_plot.set_ylabel('$\\bar{r}_E(t)$  (Hz)', labelpad=-10)
    rates, time, rec, *_ = run_sim(N, w_EE0, w_IE, w_II, w_EI0, tau_ou, tau_STDP, tau_rprim, tau_r, tau_wee, tau_wei, w_EX, w_IX, aE, aI, b, c_x, 0.0, 0.0, SIMLEN_IC, pulse_num=1)
    mask = (time >= 6.68) & (time < max_time)
    ax2.plot(rec[0, 0, :], np.abs(rec[0, 1, :]), color=clr, lw=lt - 0.5)
    ax2.scatter(rec[0, 0, -1], np.abs(rec[0, 1, -1]), color=clr, s=10, zorder=50)
    ax_plot.plot(time[mask], rates[0, mask], color=color_trace, linestyle='solid', lw=lt - 0.5, zorder=50, label='Theory')
    ax_plot.set_xlabel('Time (sec)')
    ax_plot.set_ylim(0, 80)
    ax_plot.set_xlim(6.7, 7.2)
ax2.set_title('')
ax2.set_xlabel('$w_{EE}$', labelpad=-10)
ax2.set_ylabel('$w_{EI}$', labelpad=-5)
ax2.set_xlim(0.0, 0.0035)
ax2.set_ylim(0.0, 0.004)
# ──────────────────────────────────────────────────────────────────────────
# Load spiking pulse responses for the two example locations
# ──────────────────────────────────────────────────────────────────────────
sigma = 0
seeds_all = np.arange(1)
COLORS = dict(IC1='tab:orange', IC2=colors[idxcol])
SIMLEN_IC = 10
print('loading the data')
# Load and overlay spiking responses for the same two conditions.
for key in ('IC1', 'IC2'):
    data_dir = Path('Fig5data')
    w_EE0 = WEIGHTS_RAW[key]['w_EE']
    w_EX = 20.0 / factor * tau_rprim
    w_IX = 15.0 / factor * tau_rprim
    w_IE = 10.0 / factor * tau_rprim
    w_II = 10.0 / factor * tau_rprim
    num = (-b + b * N * sc(w_EE0) + aE * N * w_EX) * (1 + N * w_II)
    den = b * N ** 2 * w_IE + aI * N ** 2 * w_IX
    w_EI0 = rev_sc(num / den)
    WEIGHTS_RAW[key]['w_EI'] = w_EI0
    w_II = np.around(w_II * factor, 3)
    w_IE = np.around(w_IE * factor, 3)
    re = load_stack('one_pulse_mean_rates_', WEIGHTS_RAW[key]['w_EE'], WEIGHTS_RAW[key]['w_EI'], SIMLEN_IC, seeds_all, weighted=True)[:, 0]
    t_ic = np.linspace(0, SIMLEN_IC, re.shape[1])
    mask = (t_ic >= 6.7) & (t_ic <= max_time)
    if key == 'IC2':
        ax_plot = ax3_IC2
    else:
        ax_plot = ax3_IC1
    clr = COLORS[key]
    ax_plot.plot(t_ic[mask], re.mean(0)[mask], color=clr, lw=lt - 0.5, linestyle=':')
    ax_plot.fill_between(t_ic[mask], re.mean(0)[mask] - re.std(0)[mask] / np.sqrt(len(seeds_all)), re.mean(0)[mask] + re.std(0)[mask] / np.sqrt(len(seeds_all)), color=clr, alpha=0.5, zorder=20, label='Sim')
    ax_plot.set_yticks([0, 90], [0, 90])
    ax_plot.legend(loc='upper right', frameon=False, fontsize=LEGEND_FS)
# ──────────────────────────────────────────────────────────────────────────
# Simulation response metrics across the line attractor
# ──────────────────────────────────────────────────────────────────────────
wEE_range = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34])
s_coord, points, p0, direction = s_coord_fun(wEE_range / scale_factor * tau_rprim, N)
max_time2 = 10.0
max_time = 7.5
wEE_range_full = np.linspace(0, wEE_range[-1] * tau_rprim / scale_factor, 100)
s_coord_full, points, p0, direction = s_coord_fun(wEE_range_full, N)
print('s_coord_full max', s_coord_full[-1])
print('s_coord max', s_coord[-1])
s_coord = (s_coord - np.min(s_coord_full)) / (np.max(s_coord_full) - np.min(s_coord_full))
print('s_coord max', s_coord[-1])
seeds_all = np.arange(1)
tau_sim = []
sim_peak_list = []
sim_var_list = []
balance_list = []
sim_energy_list = []
sim_peak_list_std = []
sim_var_list_std = []
tau_sim_std = []
balance_list_std = []
sim_energy_list_std = []
tau_r = 8.0 / ms_per_sec
data_dir = Path('Fig5data')
SIMLEN_IC = 10
tau_wee = 225000.0
tau_wei = 75000.0
sigma = 0.0
c_x = 0.0
# Compute each simulation metric at every sampled line-attractor location.
for wee in wEE_range:
    factor = np.sqrt(N)
    w_EX = 20.0 / factor * tau_rprim
    w_IX = 15.0 / factor * tau_rprim
    w_IE = 10.0 / factor * tau_rprim
    w_II = 10.0 / factor * tau_rprim
    num = (-b + b * N * sc(wee) + aE * N * w_EX) * (1 + N * w_II)
    den = b * N ** 2 * w_IE + aI * N ** 2 * w_IX
    wei = num / den / tau_rprim * factor
    w_II = np.around(w_II * factor, 3)
    w_IE = np.around(w_IE * factor, 3)
    re = load_stack('one_pulse_mean_rates_', wee, wei, SIMLEN_IC, seeds_all, weighted=True)[:, 0]
    t_ic = np.linspace(0, SIMLEN_IC, len(re[0, :]))
    balance = load_stack('one_pulse_balance_idx', wee, wei, SIMLEN_IC, seeds_all, mode='weights', weighted=True)[:, :-1]
    mask = (t_ic >= 6.7) & (t_ic <= max_time)
    mask2 = (t_ic >= 7.2) & (t_ic <= max_time2)
    t = t_ic[mask].astype(float)
    taus = []
    max_vals = []
    energy_vals = []
    for j in range(len(seeds_all)):
        y = re[j, mask].astype(float)
        max_vals.append(np.max(y) - 20)
        t_pulse = 6.8
        dt = float(np.median(np.diff(t)))
        pulse_dur = 10 * dt
        t_off = t_pulse + pulse_dur
        dt = float(np.median(np.diff(t)))
        delta_y = y - 20
        energy = np.sum(delta_y ** 2) * dt
        energy_vals.append(energy)
        peak_window_ms = 0.01
        i0 = np.searchsorted(t, t_off, side='left')
        i1 = min(t.size, i0 + int(peak_window_ms / dt))
        i_peak = i0 + int(np.argmax(y[i0:i1]))
        t_peak = t[i_peak]
        y_peak = y[i_peak]
        y_inf = 20
        amp0 = y_peak - y_inf
        target = y_inf + amp0 / np.e
        post = y[i_peak:]
        tpost = t[i_peak:]
        idx = np.where(post <= target)[0]
        if idx.size:
            j = int(idx[0])
            if j == 0:
                tau_1e = 0.0
            else:
                y1, y2 = (post[j - 1], post[j])
                t1, t2 = (tpost[j - 1], tpost[j])
                frac = (target - y1) / (y2 - y1)
                t_cross = t1 + frac * (t2 - t1)
                tau_1e = float(t_cross - t_peak)
        taus.append(tau_1e)
    dt = float(np.median(np.diff(t)))
    pre = (t_ic >= 6.5) & (t_ic < 6.75)
    mask_energy = (t_ic >= 6.8 + 10 * dt) & (t_ic <= max_time)
    baseline = np.mean(re[:, pre], axis=1, keepdims=True)
    delta_y = re - baseline
    delta_y = np.mean(delta_y, axis=0)
    energy = np.sum(delta_y[mask_energy] ** 2) * dt
    sim_energy_list.append(energy)
    y = np.mean(re[:, mask], axis=0).astype(float)
    t_off = 6.8 + 10 * dt
    peak_window_ms = 0.01
    i0 = np.searchsorted(t, t_off, side='left')
    i1 = min(t.size, i0 + int(peak_window_ms / dt))
    i_peak = i0 + int(np.argmax(y[i0:i1]))
    t_peak = t[i_peak]
    y_peak = y[i_peak]
    y_inf = 20.0
    target = y_inf + (y_peak - y_inf) / np.e
    post = y[i_peak:]
    tpost = t[i_peak:]
    tau_1e = np.nan
    idx = np.where(post <= target)[0]
    if idx.size:
        j = int(idx[0])
        if j == 0:
            tau_1e = 0.0
        else:
            y1, y2 = (post[j - 1], post[j])
            t1, t2 = (tpost[j - 1], tpost[j])
            frac = (target - y1) / (y2 - y1)
            t_cross = t1 + frac * (t2 - t1)
            tau_1e = t_cross - t_peak
    tau_sim.append(tau_1e)
    balance_list.append(np.mean(np.mean(balance[:, mask2], axis=0)))
    balance_list_std.append(np.std(np.mean(balance[:, mask2], axis=0)))
    tau_sim_std.append(np.std(taus))
    sim_peak_list.append(np.mean(max_vals))
    sim_peak_list_std.append(np.std(max_vals))
    sim_energy_list_std.append(np.std(energy_vals))
    sim_var_list.append(np.mean(np.var(re[:, mask2], axis=1)))
    sim_var_list_std.append(np.std(np.var(re[:, mask2], axis=1)))
tau_sim_std = np.array(tau_sim_std)
ax_tau.plot(s_coord, tau_sim, color='k', label='Sim', linestyle=':')
ax_tau.fill_between(s_coord, tau_sim - tau_sim_std, tau_sim + tau_sim_std, color='k', alpha=0.3)
sim_energy_list = np.array(sim_energy_list)
sim_energy_list_std = np.array(sim_energy_list_std)
ax_energy.plot(s_coord, sim_energy_list, color='k', linestyle=':')
sim_peak_list_std = np.array(sim_peak_list_std)
ax_energy.fill_between(s_coord, sim_energy_list - sim_energy_list_std, sim_energy_list + sim_energy_list_std, color='k', alpha=0.3)
ax_var.plot(s_coord, sim_var_list, color='k', linestyle=':')
sim_var_list_std = np.array(sim_var_list_std)
ax_var.fill_between(s_coord, sim_var_list - sim_var_list_std, sim_var_list + sim_var_list_std, color='k', alpha=0.3)
gain = np.array(sim_peak_list)
gain_std = np.array(sim_peak_list_std)
A_var = gain_std ** 2
V_bar = np.array(sim_var_list)
V_std = np.array(sim_var_list_std)
sigma_sigma = np.sqrt(V_bar)
sigma_var = V_std ** 2 / (4.0 * V_bar)
snr = gain / sigma_sigma
snr_var = A_var / sigma_sigma ** 2 + gain ** 2 * sigma_var / sigma_sigma ** 4
snr_std = np.sqrt(snr_var)
ax_SNR.plot(s_coord, snr, linestyle=':', color='k')
ax_SNR.fill_between(s_coord, snr - snr_std, snr + snr_std, color='k', alpha=0.3)
ax_balance.plot(s_coord, balance_list, color='k', linestyle=':')
balance_list = np.array(balance_list)
balance_list_std = np.array(balance_list_std)
ax_balance.fill_between(s_coord, balance_list - balance_list_std, balance_list + balance_list_std, color='k', alpha=0.3)

# ──────────────────────────────────────────────────────────────────────────
# Theory response metrics and autocovariance across the line attractor
# ──────────────────────────────────────────────────────────────────────────
def lags_from_freq_grid(f):
    f = np.asarray(f)
    N = f.size
    df = np.median(np.diff(f))
    T = 1.0 / df
    dt = T / N
    t = (np.arange(N) - N // 2) * dt
    return t
w_EX = 20.0 / factor * tau_rprim
w_IX = 15.0 / factor * tau_rprim
w_IE = 10.0 / factor * tau_rprim
w_II = 10.0 / factor * tau_rprim
wEE_range = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34])
wEE_range_full = np.linspace(0, wEE_range[-1] * tau_rprim / scale_factor, 100)
wEE_range = np.linspace(wEE_range[0] * tau_rprim / scale_factor, wEE_range[-1] * tau_rprim / scale_factor, 2 * len(wEE_range) - 1)
s_coord, points, p0, direction = s_coord_fun(wEE_range, N)
print('s_coord max right agter init', s_coord[-1])
s_coord_full, _, _, direction = s_coord_fun(wEE_range_full, N)
lambda_list = []
tau_list = []
amp_peak_list = []
var_list = []
beta_th_list = []
energy_th_list = []
ct = 0
# Evaluate the corresponding theory quantities along the line attractor.
for wEE, wEI in points:
    W = np.array([[wEE, -wEI], [w_IE, -w_II]])
    tau_r_vec = np.array([tau_r, tau_r * 2.0])
    J = 1.0 / tau_r_vec * (-np.eye(2) + N * W)
    eigvals = eig(J)[0]
    pulse_amp = 0.02 / tau_r
    delta_r0 = np.array([N * W[0, 0] * 20 * pulse_amp, N * W[1, 0] * 20 * pulse_amp])
    real_parts = np.real(eigvals)
    lambda_ = np.min(np.abs(real_parts))
    lambda_list.append(lambda_)
    tau_list.append(1.0 / lambda_)
    amp_peak_list.append(delta_r0[0])
    tau_E, tau_I = tau_r_vec
    A = 20.0 * pulse_amp * tau_E
    D = (1.0 - N * wEE) * (1.0 + N * w_II) + N ** 2 * wEI * w_IE
    coeff = (N * A) ** 2 / 2
    num = (N * wEI * w_IE - wEE * (1 + N * w_II)) ** 2 + tau_I / tau_E * wEE ** 2 * D
    den = D * (tau_I * (1 - N * wEE) + tau_E * (1 + N * w_II))
    energy_theory = coeff * num / den if den > 0 else np.nan
    energy_th_list.append(energy_theory)
    r_x = np.array([aE, aI])
    W = np.array([[(N - 1) * wEE, -N * wEI], [N * w_IE, -(N - 1) * w_II]])
    lambs = np.linalg.inv(np.eye(2) - W) @ (N * W_x) @ r_x
    W = np.array([[wEE, -wEI], [w_IE, -w_II]])
    lamE, lamI = (lambs[0], lambs[1])
    nu = np.array([lamE, lamI])
    sigma_rr(W, Wx, tau_r_vec, N, ext='poisson', rec='poisson', nu_ext=r_x, tau_ext=np.array([tau_r, tau_r]), nu_rec=nu, tau_rec=tau_r_vec)
    cov = calculate_CSD(N, 2, 2, N * W, W_x, tau_STDP, tau_r, tau_ou, b, aE, aI, 0, 0, lamE, lamI, normalize=True, var=True)
    var_val = cov[0, 0]
    var_list.append(cov[0, 0])
    if ct % 2 == 0:
        print('wee', wEE / tau_rprim * scale_factor)
    beta_E = abs(N * wEE * lamE - N * wEI * lamI + N * w_EX * aE) / (N * wEE * lamE + N * w_EX * aE)
    beta_th_list.append(beta_E)
    if ct % 4 == 0 and ct < 28:
        cov = calculate_CSD(N, 2, 2, N * W, W_x, tau_STDP, tau_r, tau_ou, b, aE, aI, 0, 0, lamE, lamI, normalize=True, cov=True)[0, 0]
        omega = np.linspace(-2500, 2500, 100000)
        lags = lags_from_freq_grid(omega)
        s = (s_coord - np.min(s_coord_full)) / (np.max(s_coord_full) - np.min(s_coord_full))
        ax_corr_th.plot(lags, np.real(cov), color=colors[int(wEE / wEE_range[-1] * 20 + 1)], linewidth=lt, label=f's:{s[ct]:.2f}')
        ax_corr_th.set_xlim(-0.2, 0.2)
        ax_corr_th.set_ylim(-7, 45)
        ax_corr_th.legend(loc='upper right', frameon=False, fontsize=6)
        i0 = cov.size // 2
        print('cov_th(0) =', cov[i0], 'var_th =', var_val)
    ct += 1
print('CHECK RATOP THEORY VAR', np.array(var_list[::2]) / sim_var_list)
W = np.array([[wEE, -(0.2 * wEI)], [w_IE, -w_II]])
tau_r_vec = np.array([tau_r, tau_r * 2.0])
J = 1.0 / tau_r_vec * (-np.eye(2) + N * W)
eigvals = eig(J)[0]
real_parts = np.real(eigvals)
print('real_parts', eigvals)
low_pt = np.array([sc(WEIGHTS_RAW['IC2']['w_EE']), sc(WEIGHTS_RAW['IC2']['w_EI'])]) - p0
low_s_coord = np.dot(low_pt, direction)
low_s_coord = (low_s_coord - np.min(s_coord)) / (np.max(s_coord) - np.min(s_coord))
high_pt = np.array([sc(WEIGHTS_RAW['IC1']['w_EE']), sc(WEIGHTS_RAW['IC1']['w_EI'])]) - p0
high_s_coord = np.dot(high_pt, direction)
high_s_coord = (high_s_coord - np.min(s_coord)) / (np.max(s_coord) - np.min(s_coord))
print('s_coord max', s_coord[-1])
print('s_coord_full max', s_coord_full[-1])
s_coord = (s_coord - np.min(s_coord_full)) / (np.max(s_coord_full) - np.min(s_coord_full))
print('s_coord max', s_coord[-1])
idx_high_closest = np.argmin(np.abs(s_coord - high_s_coord))
idx_low_closest = np.argmin(np.abs(s_coord - low_s_coord))
ax_tau.plot(s_coord, tau_list, linewidth=lt, color='k', label='Theory')
ax_energy.plot(s_coord, energy_th_list, linewidth=lt, color='k')
ax_energy.set_ylim(0, 160)
ax_var.plot(s_coord, var_list, linewidth=lt, color='k')
ax_var.set_ylim(-1, 80)
ax_var.set_yticks([0, 70], [0, 80])
ax_energy.scatter(low_s_coord, energy_th_list[idx_low_closest], color=colors[idxcol], s=10, zorder=20)
ax_energy.scatter(high_s_coord, energy_th_list[idx_high_closest], color='tab:orange', s=10, zorder=20)
ax_tau.scatter(low_s_coord, tau_list[idx_low_closest], color=colors[idxcol], s=10, zorder=20, label='Low')
ax_tau.scatter(high_s_coord, tau_list[idx_high_closest], color='tab:orange', s=10, zorder=20, label='High')
ax_tau.legend(loc='upper left', frameon=False)
ax_var.scatter(low_s_coord, var_list[idx_low_closest], color=colors[idxcol], s=10, zorder=20)
ax_var.scatter(high_s_coord, var_list[idx_high_closest], color='tab:orange', s=10, zorder=20)
SNR = np.array(amp_peak_list) / np.array(np.sqrt(var_list))
ax_SNR.plot(s_coord, SNR, color='k')
ax_SNR.scatter(low_s_coord, SNR[idx_low_closest], color=colors[idxcol], s=10, zorder=20)
ax_SNR.scatter(high_s_coord, SNR[idx_high_closest], color='tab:orange', s=10, zorder=20)
ax_SNR.set_ylabel('$\\delta r / \\sigma[r_E]$', labelpad=-10)
ax_SNR.set_xlabel('Location on Line')
ax_SNR.set_title('SNR')
ax_SNR.set_yticks([0, 15], [0, 15])
ax_balance.plot(s_coord, beta_th_list, color='k')
ax_balance.scatter(low_s_coord, beta_th_list[idx_low_closest], color=colors[idxcol], s=10, zorder=20)
ax_balance.scatter(high_s_coord, beta_th_list[idx_high_closest], color='tab:orange', s=10, zorder=20)
ax_balance.set_ylabel('$\\beta$', labelpad=-10)
ax_balance.set_xlabel('Location on Line')
ax_balance.set_title('Balance idx')
ax_balance.set_ylim(0, 1.0)
ax_tau.set_xlabel('Location on Line')
ax_energy.set_xlabel('Location on Line')
ax_energy.set_ylabel('Energy', labelpad=-10)
ax_tau.set_ylabel('$\\tau_{\\mathrm{decay}} $', fontsize=ft, labelpad=-10)
ax_tau.set_title('$\\tau_{\\mathrm{decay}}$ on Line', fontsize=ft)
ax_var.set_xlabel('Location on Line')
ax_var.set_ylabel('Var $r_E$ ', labelpad=-10)
ax_energy.set_title('Energy on Line ')
ax_var.set_title('Var $r_E$ on Line')
# ──────────────────────────────────────────────────────────────────────────
# Simulation autocovariance and response-trace insets
# ──────────────────────────────────────────────────────────────────────────
wEE_range = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34])
seeds_all = np.arange(1)
SIMLEN_IC = 10.0
ACF_AMPS = []
ACF_INSETS = []
ct = 1
# Load each simulation condition and estimate its autocovariance.
for i in range(len(wEE_range)):
    wee = wEE_range[i]
    sim_auto_corr = []
    factor = np.sqrt(N)
    w_EX = 20.0 / factor * tau_rprim
    w_IX = 15.0 / factor * tau_rprim
    w_IE = 10.0 / factor * tau_rprim
    w_II = 10.0 / factor * tau_rprim
    num = (-b + b * N * sc(wee) + aE * N * w_EX) * (1 + N * w_II)
    den = b * N ** 2 * w_IE + aI * N ** 2 * w_IX
    wei = num / den / tau_rprim * factor
    re = load_stack('one_pulse_mean_rates_', wee, wei, SIMLEN_IC, seeds_all, weighted=True)[:, 0]
    print('re shape', np.shape(re))
    t_ic = np.linspace(0, SIMLEN_IC, re.shape[1])
    mask2 = (t_ic >= 7.2) & (t_ic <= max_time2)
    mask2 = (t_ic >= 8.5) & (t_ic <= max_time2)
    mask3 = (t_ic >= 6.7) & (t_ic <= 7.0)
    for j in range(len(seeds_all)):
        dt = float(np.median(np.diff(t_ic)))
        lags_s, acf = acf_from_mask(re[j, :], mask2, dt, max_lag_s=0.2)
        acf_mean = acf.mean(axis=0)
        sim_auto_corr.append(acf_mean)
        acf0 = acf[0, len(lags_s) // 2]
        varj = np.var(re[j, mask2], ddof=0)
        print('sim acf(0) =', acf0, 'sim var =', varj, 'ratio =', acf0 / varj)
    lgs, acov = acf_from_mask(re, mask2, dt, max_lag_s=0.2, unbiased=False)
    acf0_seeds = acov[:, len(lgs) // 2]
    var_seeds = np.var(re[:, mask2], axis=1, ddof=0)
    print('max |acf0-var|:', np.max(np.abs(acf0_seeds - var_seeds)))
    print('mean ratio:', np.mean(acf0_seeds) / np.mean(var_seeds))
    sim_auto_corr = np.mean(np.array(sim_auto_corr), axis=0)
    if i % 2 == 0 and i < len(wEE_range) - 3:
        print('wee', wee)
        ax_corr.plot(lags_s, np.real(sim_auto_corr), color=colors[int(wee / wEE_range[-1] * 20 - 1)], linewidth=lt)
    ax_corr.set_xlim(-0.2, 0.2)
    ax_corr.set_ylim(-7, 45)
    ax_corr.set_yticks([0, 40], [0, 40])
    ax_corr.set_xticks([-0.2, 0, 0.2], [-0.2, 0, 0.2])
    ax_corr_th.set_xticks([-0.2, 0, 0.2], [-0.2, 0, 0.2])
    ax_corr.set_xlabel('Lag (s)')
    ax_corr.set_ylabel('$C(\\tau)$', labelpad=-10)
    ax_corr.set_title('Sim Autocorrelation')
    ax_corr_th.set_title('Theory Autocorrelation')
    ax_corr_th.set_ylabel('$C(\\tau)$', labelpad=-10)
    ax_corr_th.set_xlabel('Lag (s)')
    ax_corr_th.set_yticks([0, 40], [0, 40])
    if 'points' not in locals():
        _w = np.linspace(0, w_EE_vals[-1], 200)
        _num = (-b + b * N * _w + aE * N * w_EX) * (1 + N * w_II)
        _den = b * N ** 2 * w_IE + aI * N ** 2 * w_IX
        _wEI = _num / _den
        points = np.column_stack((_w, _wEI))
    x0 = sc(wee)
    y0 = (-b + b * N * x0 + aE * N * w_EX) * (1 + N * w_II) / (b * N ** 2 * w_IE + aI * N ** 2 * w_IX)
    x_min, x_max = ax2.get_xlim()
    y_min, y_max = ax2.get_ylim()
    dx = 0.6 * (x_max - x_min)
    dy = 0.2 * (y_max - y_min)
    xa = x0 - dx
    ya = y0 + dy
    xy_disp = ax2.transData.transform([xa, ya])
    xy_fig = fig.transFigure.inverted().transform(xy_disp)
    w_frac, h_frac = (0.08 * (wee / 100), 0.5)
    left = xy_fig[0] - w_frac / 2
    bottom = xy_fig[1] - h_frac / 2
    iax = fig.add_axes([left, bottom, w_frac, h_frac], zorder=30, facecolor='none')
    for sp in iax.spines.values():
        sp.set_visible(False)
    iax.set_xticks([])
    iax.set_yticks([])
    denom = wEE_range.max() - wEE_range.min()
    c_idx = 0 if denom == 0 else int(np.clip((wee - wEE_range.min()) / denom * (len(colors) - 3), 0, len(colors) - 3))
    _col = colors[c_idx]
    lag_lim = 0.017 * (1 + ct / len(wEE_range)) ** 4
    print('lag_lim', lag_lim)
    idx = np.where(np.abs(lags_s) < lag_lim)
    if i < len(wEE_range) - 2:
        print('shape tic', np.shape(t_ic), 'shape re', np.shape(re), 'shape mask', np.shape(mask3))
        iax.plot(t_ic[mask3], np.mean(re, axis=0)[mask3], linewidth=max(0.7 * lt, 0.5), color=_col, alpha=0.95, zorder=35 - ct)
        iax.set_xlim(6.75, 7)
    amp = float(np.nanmax(np.abs(sim_auto_corr[idx])))
    ACF_AMPS.append(amp)
    ACF_INSETS.append(iax)
    ct += 1
Y = 1.1 * np.nanmax(ACF_AMPS)
for a in ACF_INSETS:
    a.set_ylim(-Y, Y)
ax_corr.legend(frameon=False, fontsize=6)
# ──────────────────────────────────────────────────────────────────────────
# Final styling and output
# ──────────────────────────────────────────────────────────────────────────
for ax in (ax2, ax3_IC1, ax3_IC2, ax_energy, ax_tau, ax_var, ax_SNR, ax_corr_th, ax_corr, ax_balance):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(lt)
    ax.spines['bottom'].set_linewidth(lt)
ax_tau.set_yticks([0, 0.15], [0, 0.15])
ax_energy.set_yticks([0, 500], [0, 500])
ax_energy.set_ylim(-1, 500)
ax_balance.set_yticks([0.4, 1], [0.4, 1])
TITLE_FS = 9
LABEL_FS = 9
TICK_FS = 7
LEGEND_FS = 8
plt.rcParams.update({'axes.titlesize': TITLE_FS, 'axes.labelsize': LABEL_FS, 'xtick.labelsize': TICK_FS, 'ytick.labelsize': TICK_FS, 'legend.fontsize': LEGEND_FS})
axs = [ax_tau, ax_energy, ax_var, ax_SNR, ax_balance]
for ax in axs:
    ax.set_xticks([0, 1], [0, 1])
plt.subplots_adjust(hspace=0.4, bottom=0.1, top=0.93, left=0.05, wspace=0.4, right=0.96)
plt.savefig('Fig5.pdf', dpi=150)
plt.show()
