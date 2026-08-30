# Figure 5: line-attractor regimes, stability, and convergence basins
# Simulation, analysis, and plotting calculations are retained unchanged.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as mticker
# -----------------------------------------------------------------------------
# Figure-wide font settings
# -----------------------------------------------------------------------------
TITLE_FS  = 11   # Titles
LABEL_FS  = 11   # Axis labels
TICK_FS   = 8    # Tick labels
LEGEND_FS = 9    # Legends

import matplotlib as mpl
mpl.rcParams.update({
    # The later assignment sets the active font family.
    "font.family": "sans-serif",
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}"})

plt.rcParams.update({
    "axes.titlesize": TITLE_FS,
    "axes.labelsize": LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": LEGEND_FS,
})

# -----------------------------------------------------------------------------
# Model and simulation parameters
# -----------------------------------------------------------------------------
lt = 1.5

N = 500
aE = 20
aI = aE
# tau_rprim is an irrelevant scale factor to allow the weights to be written in integers
tau_rprim = 2.0 / 1000.0
scale_factor = np.sqrt(N)
w_II  = 10.0 / scale_factor * tau_rprim
w_EE0 = 15.0 / scale_factor * tau_rprim        # initial w_EE  (simulation trajectory)
w_EI0 = 13.0 / scale_factor * tau_rprim        # initial w_EI

#w_EE0 = 11.0 / scale_factor * tau_rprim        # initial w_EE  (simulation trajectory)
#w_EI0 = 18.0 / scale_factor * tau_rprim  
b     = 20.0

ms_per_sec = 1000.0

tau_STDP = 120.0 / ms_per_sec          # Plasticity time scale
w_IX = 15.0 / scale_factor * tau_rprim
w_EX = 20.0 / scale_factor * tau_rprim
Wx   = np.array([[w_EX, 0.0], [0.0, w_IX]], dtype=float)
rx   = np.array([aE, aI])

# Time scales (s)
tau_r    = 8./1000.0           # neuronal rate decay (smaller → faster rates)
tau_re = tau_r
tau_ri = tau_r*2
tau_w_e  = 75000
tau_w_i  = 225000
s_sqrt   = np.sqrt(tau_w_e / tau_w_i)

# Simulation horizon (s)
sim_duration = 60.0

# -----------------------------------------------------------------------------
# Analytic line-attractor and fixed-rate helpers
# -----------------------------------------------------------------------------
def wEI_line_fun(wEE,w_IE,w_II,w_IX,aE,aI,b):
    num = (-b + b*N*wEE + aE*N*w_EX) * (1 + N*w_II)
    den = (b*N**2*w_IE + aI*N**2*w_IX)
    return num / den

def wEI_sep(wEE,w_IE,w_II,w_EX,aE,aI):
    num = (aE * w_EX * (1 + N * w_II) - s_sqrt * aE * N * w_EX * w_IE - s_sqrt * aI * w_IX * (1 - N * wEE))
    den = aI * N * w_IX
    return num / den

def condition(re_fixed):
    return re_fixed > 0

def condition2(w_ei, wEIval_rEzero):
    return w_ei > (wEIval_rEzero)

# Closed-form rate functions retained for completeness
def R(wEE, wEI, w_IE):
    num = N * (-aE * (w_EX + N * w_EX * w_II) + aI * N * wEI * w_IX)
    den = -1 + N * (wEE - w_II) + N**2 * (-wEI * w_IE + wEE * w_II)
    return num / den

def Q(wEE, wEI, w_IE):
    num = N * (aE * N * w_EX * w_IE + aI * (1 - N * wEE) * w_IX)
    den = 1 + N * (-wEE + w_II) + N**2 * (wEI * w_IE - wEE * w_II)
    return num / den

# Vector-field, stability, and line-coordinate calculations
def solve_rates(wEE, wEI):
    W = np.array([[wEE, -wEI],
                  [w_IE, -w_II]], dtype=float)
    M = np.eye(2) - N * W
    v = N * (Wx @ rx)
    try:
        r = np.linalg.solve(M, v)
        return float(r[0]), float(r[1])
    except np.linalg.LinAlgError:
        return np.nan, np.nan

def VF_homeo(w):
    wEE, wEI = float(w[0]), float(w[1])
    rE, rI = solve_rates(wEE, wEI)
    g = (rE - b)
    return np.array([(tau_STDP / tau_w_e) * rE * g,
                     (tau_STDP / tau_w_i) * rI * g], dtype=float)

def rk4_streamline_unbounded(w0, step_len=1e-4, n_steps=20000, direction=+1):
    """Unbounded arc-length RK4: dw/ds = direction * F/||F||."""
    path = [np.array(w0, dtype=float)]
    w = np.array(w0, dtype=float)

    def G(x):
        Fx = VF_homeo(x)
        n  = np.linalg.norm(Fx)
        if n == 0 or not np.isfinite(n):
            return np.array([np.nan, np.nan])
        return (direction * Fx / n)

    for _ in range(n_steps):
        k1 = G(w)
        if not np.isfinite(k1).all(): break
        k2 = G(w + 0.5 * step_len * k1)
        k3 = G(w + 0.5 * step_len * k2)
        k4 = G(w + step_len * k3)
        w  = w + (step_len / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(w).all(): break
        path.append(w.copy())
    return np.asarray(path)

def jacobian_F(w, eps=1e-7):
    f0 = VF_homeo(w)
    J = np.zeros((2, 2), dtype=float)
    for j in range(2):
        e = np.zeros(2); e[j] = eps
        f1 = VF_homeo(w + e)
        J[:, j] = (f1 - f0) / eps
    return J

def lambda_on_line_closed_form(N, w_EE, w_IE, w_II, w_EX, w_IX, aE, aI, b, tau_w_e, tau_w_i):
    """
    Closed-form nonzero eigenvalue along the line attractor r_E = b.
    Supports scalars or numpy arrays.
    """
    N      = np.asarray(N)
    w_EE   = np.asarray(w_EE);   w_IE = np.asarray(w_IE);  w_II = np.asarray(w_II)
    w_EX   = np.asarray(w_EX);   w_IX = np.asarray(w_IX)
    aE     = np.asarray(aE);     aI   = np.asarray(aI)
    b      = np.asarray(b)
    tau_w_e = np.asarray(tau_w_e); tau_w_i = np.asarray(tau_w_i)

    num_left = N * (b * w_IE + aI * w_IX)

    quad = (
        b**2 * tau_w_i
        - b**2 * N**2 * tau_w_e * w_IE**2
        + 2 * b**2 * N * tau_w_i * w_II
        + b**2 * N**2 * tau_w_i * w_II**2
        - 2 * aI * b * N**2 * tau_w_e * w_IE * w_IX
        - (aI**2) * N**2 * tau_w_e * w_IX**2)

    denom = (
        tau_w_e * tau_w_i * (1 + N * w_II)**2
        * (aE * N * w_EX * w_IE + aI * w_IX - aI * N * w_EE * w_IX)
    )

    return (num_left * quad) / denom

def s_coord_fun(wEE_range, N):
    # Compute wEI analytically
    num = (-b + b * N * wEE_range + aE * N * w_EX) * (1 + N * w_II)
    den = (b * N**2 * w_IE + aI * N**2 * w_IX)
    wEI_line = num / den
    # Build (wEE, wEI) pairs
    points = np.column_stack((wEE_range, wEI_line))
    # Parametrize along the line
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
    s_coord = (s_coord - np.min(s_coord)) / (np.max(s_coord) - np.min(s_coord))
    return s_coord, points, p0, direction
# Inactive earlier convergence implementation retained unchanged.
"""
def convergence_to_attractor(wEE_init, wEI_init, w_IE, threshold=.2e-4): 
    Run a simulation from initial (wEE, wEI) and return True if it ends up near the line attractor. 
    t_sim, W_sim = run_sim_short(N, wEE_init, w_IE, w_II, wEI_init, tau_STDP, tau_rprim, tau_r, tau_w_e, tau_w_i, w_EX, w_IX, aE, aI, b, sim_duration) 
    wEE_final = W_sim[0, 0] 
    wEI_final = np.abs(W_sim[0, 1] )
    wEI_target = wEI_line_fun(wEE_final, w_IE, w_II, w_IX, aE, aI, b) 
    print('wEI_final',wEI_final,'vwEI_target',wEI_target)
    return abs(wEI_final - wEI_target) < threshold
"""

# Classify whether an initial condition reaches the line attractor.
def convergence_to_attractor(wEE0, wEI0, w_IE,
                             tol_dist=2e-5, tol_rE=5e-3):
    # same args, same order, same duration as everywhere else
    args = (N, wEE0, w_IE, w_II, wEI0,
            tau_STDP, tau_rprim, tau_re,tau_ri,
            tau_w_e, tau_w_i,
            w_EX, w_IX, aE, aI, b,
            sim_duration,tol_dist)
    t, W = run_sim_short(*args)

    # final weights in magnitude coordinates (consistent with quiver & separatrix)
    wEEf = float(np.abs(W[0, 0]))
    wEIf = float(np.abs(W[0, 1]))

    # 1) geometric closeness to your line attractor
    d = abs(wEIf - wEI_line_fun(wEEf, w_IE, w_II, w_IX, aE, aI, b))
    if d > tol_dist:
        return False

    # 2) actually on the rE≈b line (not just geometrically close)
    rE, rI = solve_rates(wEEf, wEIf)
    if (not np.isfinite(rE)) or abs(rE - b) > tol_rE:
        return False

    return True

# -----------------------------------------------------------------------------
# Plasticity simulations
# -----------------------------------------------------------------------------
def run_sim(N, w_EE, w_IE, w_II, w_EI,
            tau_STDP, tau_rprim, tau_re,tau_ri,
            tau_w_e, tau_w_i,
            w_EX, w_IX, aE, aI, b,
            sim_T):

    ms_per_sec = 1000.0
    dt   = 0.1 / ms_per_sec
    time = np.linspace(0, sim_T, int(sim_T / dt))

    dwdt  = np.zeros((2, 2, len(time)))
    W_rec = np.zeros((2, 2, len(time)))
    rates = np.zeros((2, len(time)))

    # External-input matrix
    Wx = np.diag([w_EX, w_IX])

    # Initial recurrent weights
    W_rec[:, :, 0] = np.array([[w_EE, -w_EI],
                               [w_IE, -w_II]])

    for k in range(len(time) - 1):
        rE, rI = rates[:, k]
        r_vec  = np.array([rE, rI])
        Wk     = W_rec[:, :, k]

        tau_r_vec = np.array([tau_re, tau_ri])
        drdt = (-r_vec / tau_r_vec + (N / tau_r_vec) * (Wk @ r_vec + Wx @ np.array([aE, aI])))

        post_neurons = (rE - b)
        dwdt[0, 0, k] = (1 / tau_w_e) * (tau_STDP * rE * post_neurons)
        dwdt[0, 1, k] = -(1 / tau_w_i) * (tau_STDP * rI * post_neurons)

        # Euler updates
        W_rec[:, :, k + 1] = Wk + dwdt[:, :, k] * dt
        rates[:, k + 1]    = r_vec + drdt * dt

    print('rates',rates[:,-5])
    print('W_rec',W_rec[:,:,-5])
    return time, np.abs(W_rec)          # positive weights for plotting


def run_sim_short(N, w_EE, w_IE, w_II, w_EI,
            tau_STDP, tau_rprim, tau_re,tau_ri,
            tau_w_e, tau_w_i,
            w_EX, w_IX, aE, aI, b,
            sim_T,tol_dist):

    ms_per_sec = 1000.0
    #dt   = 0.1 / ms_per_sec
    dt   = 0.1 / ms_per_sec
    time = np.linspace(0, sim_T, int(sim_T / dt))

    dwdt  = np.zeros((2, 2))
    W_rec = np.zeros((2, 2))
    rates = np.zeros((2))

    # external-input matrix
    Wx = np.diag([w_EX, w_IX])

    # initial recurrent weights
    W_rec[:, :] = np.array([[w_EE, -w_EI],
                               [w_IE, -w_II]])

    for k in range(len(time) - 1):
        rE, rI = rates
        r_vec  = np.array([rE, rI])
        Wk     = W_rec[:, :]

        tau_r_vec = np.array([tau_re, tau_ri])
        drdt = (-r_vec / tau_r_vec + (N / tau_r_vec) * (Wk @ r_vec + Wx @ np.array([aE, aI])))

        post_neurons = (rE - b)
        dwdt[0, 0] = (1 / tau_w_e) * (tau_STDP * rE * post_neurons)
        dwdt[0, 1] = -(1 / tau_w_i) * (tau_STDP * rI * post_neurons)

        # Euler updates
        W_rec[:, :] = Wk + dwdt[:, :] * dt
        rates[:]    = r_vec + drdt * dt
        if np.any(np.abs(W_rec[:, :]) > 2) or np.any(rates[:] ) > 100:
            print('time',time[k],'return weights',np.abs(W_rec[:, :]),'return rates,',rates[:] )
            return time, np.abs(W_rec) 
        
        wEEf = float(np.abs(W_rec[0, 0]))
        wEIf = float(np.abs(W_rec[0, 1]))
        # 1) geometric closeness to your line attractor
        #d = abs(wEIf - wEI_line_fun(wEEf, w_IE, w_II, w_IX, aE, aI, b))
        #if d < tol_dist:
        #    return time, np.abs(W_rec) 

    return time, np.abs(W_rec)   


# -----------------------------------------------------------------------------
# Bottom-row phase-plane plotting
# -----------------------------------------------------------------------------
def placeholder_full_subplot(ax, aE, w_IE, label, w_IE_vals ):
    # Nullclines and rate-validity boundaries
    wEEval_rIzero = (aE * N * w_EX * w_IE + aI * w_IX) / (aI * N * w_IX)
    wEIval_rEzero = (aE * w_EX * (1 + N * w_II)) / (aI * N * w_IX)
    ax.axvline(wEEval_rIzero, color='k', linestyle=':', linewidth=lt)
    ax.axhline(wEIval_rEzero, color='k', linestyle=':', linewidth=lt)

    # Weight grid for the vector field
    max_w, ngrid = 80, 15
    w_EE_vals = np.linspace(0, max_w, ngrid) / scale_factor * tau_rprim
    w_EI_vals = np.linspace(0, max_w, ngrid) / scale_factor * tau_rprim
    w_ee, w_ei = np.meshgrid(w_EE_vals, w_EI_vals, indexing='xy')

    # Fixed rates from the closed-form 2 x 2 solve

    det =(1 - N * w_ee) * (1 + N * w_II) - (N * w_ei)* (-N * w_IE)
    with np.errstate(divide='ignore', invalid='ignore'):
        re_fixed = ((1 + N * w_II) * (N * aE * w_EX) - (N * w_ei) * (N * aI * w_IX)) / det
        ri_fixed = (-(-N * w_IE) * (N * aE * w_EX) + (1 - N * w_ee) * (N * aI * w_IX)) / det

    # Normalize and mask the plasticity vector field
    dwee = re_fixed * (re_fixed - b) * tau_STDP / tau_w_e #* tau_rprim
    dwei = ri_fixed * (re_fixed - b) * tau_STDP / tau_w_i #* tau_rprim

    valid = condition(re_fixed) & np.isfinite(re_fixed) & np.isfinite(ri_fixed)

    U = np.full_like(w_ee, np.nan, dtype=float)
    V = np.full_like(w_ei, np.nan, dtype=float)
    U[valid] = dwee[valid]
    V[valid] = dwei[valid]

    norm = np.hypot(U, V)
    U = np.divide(U, norm, out=np.full_like(U, np.nan), where=norm > 0)
    V = np.divide(V, norm, out=np.full_like(V, np.nan), where=norm > 0)

    U = np.ma.masked_invalid(U)
    V = np.ma.masked_invalid(V)

    ax.quiver(w_ee, w_ei, U, V,
              scale=8., width=0.005, scale_units='inches',
              color='k', alpha=.8, zorder=10)

    # Simulated plasticity trajectory
    t_sim, W_sim = run_sim(N, w_EE0, w_IE, w_II, w_EI0,
                           tau_STDP, tau_rprim, tau_re,tau_ri,
                           tau_w_e, tau_w_i, w_EX, w_IX, aE, aI, b, sim_duration)
    wEE_traj = W_sim[0, 0, :]
    wEI_traj = W_sim[0, 1, :]

    # Analytic curves evaluated over wEE
    wEE_range = np.linspace(-1, w_EE_vals[-1] + 0.01, 50000)

    # Line attractor
    num = (-b + b * N * wEE_range + aE * N * w_EX) * (1 + N * w_II)
    den = (b * N**2 * w_IE + aI * N**2 * w_IX)
    wEI_line = num / den
    ax.plot(wEE_range, wEI_line, color='b', linewidth=lt)

    # Non-oscillatory stability boundary + region fill
    nonosc = (-1 + N * wEE_range - N * w_II + N**2 * wEE_range * w_II) / (N**2 * w_IE)
    ax.plot(wEE_range, nonosc, color='b', linestyle='--', linewidth=2)
    osc = (tau_re+tau_ri+N*tau_re*w_II)/(N*tau_ri)
    print('osc,',osc)
    ax.axvline(osc,linestyle='dashed',linewidth=lt,color='k')
    #wEEval_rIzero = (aE * N * w_EX * w_IE + aI * w_IX) / (aI * N * w_IX)
    #wEIval_rEzero = (aE * w_EX * (1 + N * w_II)) / (aI * N * w_IX)
    idx = (wEE_range < wEEval_rIzero) & (wEE_range < osc)

    ax.fill_between(wEE_range[idx], nonosc[idx], wEIval_rEzero, color='m', alpha=.25)

    # Overlay the simulated trajectory
    ax.plot(wEE_traj, wEI_traj, lw=lt, color='k', label='Simulation')
    ax.scatter(wEE_traj[0], wEI_traj[0], s=50, marker='*', color='k', zorder=15, label='Start')

    # Separatrix (± branches)
    rad = (aI**2 * N**2 * tau_w_e * tau_w_i * w_IX**2
           * (aE * N * w_EX * w_IE + aI * (1 - N * wEE_range) * w_IX)**2)
    base = (aE * aI * N * tau_w_i * w_EX * w_IX
            + aE * aI * N**2 * tau_w_i * w_EX * w_II * w_IX)
    den_sep = (aI**2 * N**2 * tau_w_i * w_IX**2)
    sep_minus = (base - np.sqrt(rad)) / den_sep
    sep_plus  = (base + np.sqrt(rad)) / den_sep
    if w_IE== w_IE_vals[-1]:
        lst=':'
        cr='k'
    else:
        lst='solid'
        cr='tab:green'
    sep_minus_left= sep_minus[idx]
    sep_minus_right= sep_minus[~idx]
    ax.plot(wEE_range[idx], sep_minus_left, color=cr, lw=lt, label='Separatrix (analytic)',linestyle=lst)
    ax.plot(wEE_range[~idx], sep_minus_right, color='tab:green', lw=lt, label='Separatrix (analytic)')
    ax.plot(wEE_range, sep_plus,  color='tab:green', lw=lt)

    # Panel limits, ticks, and labels
    ax.set_xlim(0, 0.003)
    ax.set_ylim(0, 0.003)

    t1, t2, t3, t4 = 1e-3,2e-3, 3e-3, 4e-3
    t5 = float(wEEval_rIzero if (ax == axs[1, 1]) else wEEval_rIzero + 5e-5)
    ax.text(t5,-5e-4,r'$r_I^\ast=0$',fontsize=TICK_FS)
    if ax == axs[1, 2]:
        xticks  = [0,t1 , t4]
        xlabels = ['0','.001', '.004']
    elif ax == axs[1, 0]:
        xticks  = [0,t1 , t3]
        xlabels = ['0','.001', '.003']
    else:
        xticks  = [0,t1, t4]
        xlabels = ['0','.001', '.004']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=TICK_FS)

    if ax == axs[1, 0]:
        ax.set_ylim(0,wEIval_rEzero+2e-4)
        ax.set_yticks([0,0.001,.003])
        ax.set_yticklabels(['0','0.001','.003'], fontsize=LABEL_FS)
    elif ax == axs[1, 2]:    
        ax.set_yticks([ ])
        ax.set_ylim(0,wEIval_rEzero+2e-4)
        ax.set_yticklabels([], fontsize=LABEL_FS)

        ax.text(.9,                       # x in *axes* coords (a bit to the right of the axis)
        wEIval_rEzero-5e-4,              # y in *data* coords
        r'$r_E^* = 0$', 
        transform=ax.get_yaxis_transform(),  # (axes-x, data-y)
        va='center',
        ha='left',
        fontsize=TICK_FS,)

    else:
        ax.set_yticks([])
        ax.set_ylim(0,wEIval_rEzero+2e-4)
        ax.set_yticklabels([])

    ax.scatter(wEEval_rIzero, np.abs(wEIval_rEzero), marker='*', color='r', s=50, zorder=50)
    ax.set_xlim(0,.0048)
    ax.set_xlabel(r'$w_{EE}$', fontsize=LABEL_FS, labelpad=-3)
    if ax == axs[1, 0]:
        ax.set_ylabel(r'$w_{EI}$', fontsize=LABEL_FS, labelpad=0)
    ax.set_title(label, fontsize=TITLE_FS)

    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)

    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    ax.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)

# -----------------------------------------------------------------------------
# Build Figure 5
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(7.5, 4.5))

# Compare the three w_IE regimes across the figure.
w_IE_vals   = np.array([22, 10, 2.]) / scale_factor * tau_rprim
#w_IE_vals   = np.array([22]) / scale_factor * tau_rprim
#w_IE_vals   = np.array([20]) / scale_factor * tau_rprim
#w_IE_vals   = np.array([7.]) / scale_factor * tau_rprim
titles1     = ['Regime 1 : Full Basin ',
               'Regime 2: Partial Basin\n' + r'$b < b_{max}$',
               'Regime 2: No Basin:\n' + r'$b > b_{max}$']
lamb_labels = [r'$\lambda_{R1}$', r'$\lambda_{R2}: b<b_{max}$', r"$\lambda_{R2}:b>b_{max}$"]
colors_seq  = ['firebrick', 'mediumvioletred', 'rebeccapurple']
run_new=True  # Load precomputed convergence grids by default.

# Populate each regime column and its corresponding top-row summaries.
for i, w_IE in enumerate(w_IE_vals):
    print('============================== NEW WIE ==================================== ')
    wie  = w_IE * scale_factor
    label = rf'$\frac{{{wie:.1e}}}{{\sqrt{{N}}}}$'
    placeholder_full_subplot(axs[1, i], aE, w_IE, label=titles1[i], w_IE_vals=w_IE_vals )


    # Weight grid used for the convergence-basin overlay
    wEE_grid = np.linspace(0, 0.005, 75)
    wEI_grid = np.linspace(0, 0.004,75)
    axs[0, 1].scatter(w_IE,20,zorder=20,color=colors_seq[i])
    grid_x, grid_y = np.meshgrid(wEE_grid, wEI_grid)

    
    # Load or regenerate the convergence classification for this regime.
    DATA_DIR = "Fig5data/"
    os.makedirs(DATA_DIR, exist_ok=True)
    fname = f'data_colorgrid_{np.around(w_IE / tau_rprim * np.sqrt(N),0)}'
    csv_path = os.path.join(DATA_DIR, f"{fname}.csv")        
    if run_new:
        # Run the existing convergence simulations over the full grid.
        color_grid = np.zeros_like(grid_x)
        for m in range(grid_x.shape[0]):
            for n in range(grid_x.shape[1]):
                wEE_init = grid_x[m, n]
                wEI_init = grid_y[m, n]
                #did_converge = convergence_to_attractor(wEE_init, wEI_init, w_IE)
                did_converge= 1.0 if convergence_to_attractor(wEE_init, wEI_init, w_IE) else 0.0
                color_grid[m, n] = 1 if did_converge else 0
        
        # Save weight coordinates as row and column labels.
        pd.DataFrame(color_grid, index=wEI_grid, columns=wEE_grid).to_csv(
            csv_path, index_label="wEI")
    else:
        color_grid = pd.read_csv(csv_path, index_col="wEI").values.astype(np.uint8)
        print('color_grid',color_grid)
    
    
    wEEval_rIzero = (aE * N * w_EX * w_IE + aI * w_IX) / (aI * N * w_IX)
    wEE_axis = np.linspace(0, 2 * wEEval_rIzero, 200)

    s, _, _, _ = s_coord_fun(wEE_axis, N)
    lambs = lambda_on_line_closed_form(N, wEE_axis, w_IE, w_II, w_EX, w_IX, aE, aI, b, tau_w_e, tau_w_i)
    if i == 0 or i == 2 or i == 1:
        axs[0, 2].plot((s * 2)[:-100], lambs[:-100], label=lamb_labels[i], color=colors_seq[i])
        axs[0, 2].set_xlabel('Location on Line', fontsize=LABEL_FS, labelpad=-5)
        if i == 2:
            axs[0, i].set_title('Eigenvalue Along Line', fontsize=TITLE_FS)
        axs[0, 2].axhline(0, color='k', linestyle=':')

    if i == 0:
        # Top-left panel: b_max over (w_IE, w_II).
        w_II_grid = np.linspace(0, 30, 800) / scale_factor * tau_rprim
        w_IE_grid = np.linspace(0, 40, 700) / scale_factor * tau_rprim
        axs[0, 0].plot(w_IE_grid,(np.sqrt(tau_w_e/tau_w_i)*w_IE_grid*N-1)/N,linestyle='dashed',color='k',zorder=50)
        w_ie, w_ii = np.meshgrid(w_IE_grid, w_II_grid)
        maxval = 40 / scale_factor * tau_rprim

        # Top-middle panel: cross-section at fixed w_II.
        cross_idx = len(w_II_grid) // 2
        w_ii_val = w_II_grid[cross_idx]          # fixed
        w_ie_vals = w_IE_grid                    # x-axis

        ratio = np.sqrt(tau_w_i / tau_w_e)
        denom_cross =  N * w_ie_vals - ratio * (1 + N * w_ii_val)

        cross_values = -w_IX * aI / denom_cross * N 
        cross_values[denom_cross > 0] = maxval   # cap in Regime 1 (denom>0)
        idx=np.where(cross_values>100)
        cross_values[idx]=100

        positive_mask = denom_cross > 0
        # Boundary: N*w_IE = sqrt(tau_w_i/tau_w_e)*(1+N*w_II)
        w_ie_boundary = ratio * (1 + N * w_ii_val) / N

        axs[0, 1].set_ylim(0, 13)
        axs[0, 1].fill_between(w_ie_vals, 0, 100, where=positive_mask, color='tab:blue',  alpha=0.5, label='Regime 1')
        axs[0, 1].fill_between(w_ie_vals, 0, cross_values, where=~positive_mask, color='tab:orange', alpha=0.5, label='Regime 2')
        #axs[0, 0].axvline(w_ie_boundary, color='k', linestyle=':', label=r'$\sqrt{\frac{\tau_{w_{EE}}}{\tau_{w_{EI}}}}Nw_{IE}= 1+Nw_{II}$')
        axs[0, 1].axvline(w_ie_boundary, color='k', linestyle=':')
        axs[0, 1].plot(w_ie_vals[:np.argmax(positive_mask) if np.any(positive_mask) else 0],
                       cross_values[:np.argmax(positive_mask) if np.any(positive_mask) else 0],
                       color='k', label=r'$b_{max}$')

        # Regime 1 has unbounded b; Regime 2 has finite b_max.
        denom_2d = N * w_ie - ratio * (1 + N * w_ii)

        b_max_map = np.full_like(denom_2d, np.nan, dtype=float)
        bounded_mask_2d = denom_2d < 0
        # Evaluate finite b_max only in Regime 2.
        b_max_map[bounded_mask_2d] = -N * w_IX * aI / denom_2d[bounded_mask_2d]

        # Mask Regime 1 so it receives one uniform color.
        b_max_masked = np.ma.masked_invalid(b_max_map)
        cmap_b = plt.cm.viridis.copy()
        # Uniform color for unbounded b in Regime 1.
        cmap_b.set_bad((0.9, 0.9, 0.9, 1.0))

        im = axs[0, 0].pcolormesh(w_IE_grid,
            w_II_grid,
            b_max_masked,
            shading='auto',
            cmap=cmap_b,vmax=50,vmin=0)
        cbar = fig.colorbar(im, ax=axs[0, 0])
        cbar.set_ticks([0, 25,50])
        cbar.ax.set_yticklabels([0,'25',r'$\infty$'])

        axs[0, 0].set_xlabel(r'$w_{IE}$', fontsize=LABEL_FS, labelpad=-3)
        axs[0, 0].set_ylabel(r'$w_{II}$', fontsize=LABEL_FS, labelpad=-5)
        axs[0, 0].set_title(r'$b_{max}(w_{IE}, w_{II})$', fontsize=TITLE_FS)
        axs[0, 0].set_yticks([0,.001,.002],['0','.001','.002'])
        axs[0, 0].set_xticks([0,.001,.003],['0','.001','.003'])
        axs[0, 0].set_ylim(0,w_II_grid[-1])
    axs[0, 0].scatter(w_IE,w_II,color=colors_seq[i])
    axs[0, 0].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    axs[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    axs[0, 0].xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    axs[0, 0].ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
    axs[0, 0].yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    axs[0, 0].xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    
    #axs[0,2].pcolormesh(w_ie_vals,w_ii_val,)
    # Bottom-row convergence-basin overlay
    wEE_grid, wEI_grid = grid_x, grid_y
    # Mask oscillatory and nonphysical regions.
    w_EI_nonosc = (-1 + N * wEE_grid - N * w_II + N**2 * wEE_grid * w_II) / (N**2 * w_IE)
    osc = (tau_re+tau_ri+N*tau_re*w_II)/(N*tau_ri)
    mask_outside_nonosc = (wEI_grid < w_EI_nonosc) 
    mask_outside_osc=  (wEE_grid>osc)

    # Mask the region above the r_E = 0 boundary.
    wEIval_rEzero = (aE * w_EX * (1 + N * w_II)) / (aI * N * w_IX)
    mask_above_rEz = (wEI_grid > wEIval_rEzero) 
    white_region_mask = np.logical_or(mask_outside_nonosc, mask_above_rEz)
    
    final_color_grid = np.array(color_grid, dtype=float)
    final_color_grid[white_region_mask] = np.nan
    final_color_grid[mask_outside_osc]=np.nan
    C = np.ma.masked_invalid(final_color_grid)  # treat NaNs as missing
    cmap = ListedColormap([(1.0, 0.4, 0.4, 0.4), (0.2, 0.4, 1.0, 0.6)])
    cmap.set_bad((1, 1, 1, 0)) 
    axs[1, i].pcolormesh(grid_x, grid_y, final_color_grid, cmap=cmap, vmin=0, vmax=1, shading='auto')
    

# Legends
ax = axs[0, 2]
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles, labels, handlelength=0, frameon=False, loc=(.1, .2), fontsize=LEGEND_FS)
for txt, h in zip(leg.get_texts(), handles):
    c = getattr(h, 'get_color', lambda: None)() or getattr(h, 'get_facecolor', lambda: [None])()[0]
    if c is not None:
        txt.set_color(c)

handles, labels = axs[0, 0].get_legend_handles_labels()
legs = axs[0, 1].legend(handlelength=1, frameon=False, fontsize=LEGEND_FS, loc=(.03, .6))
for txt, h in zip(legs.get_texts(), handles):
    c = getattr(h, 'get_color', lambda: None)() or getattr(h, 'get_facecolor', lambda: [None])()[0]
    if c is not None:
        txt.set_color(c)

# Top-row labels, limits, and ticks
axs[0, 1].set_ylabel('Maximum b', fontsize=LABEL_FS, labelpad=0)
axs[0, 1].set_title('Two Regimes', fontsize=TITLE_FS)
axs[0, 1].set_xlabel(r'$w_{IE}$', fontsize=LABEL_FS, labelpad=-5)
axs[0, 1].set_yticks([0, 50], [0, 50], fontsize=TICK_FS)
axs[0, 1].set_xlim(0., .001)
axs[0, 1].set_xticks([0, .003], [0, '.003'], fontsize=TICK_FS)
axs[0, 1].set_ylim(0,70)
axs[0, 1].xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
axs[0, 1].xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
axs[0, 1].ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)

axs[0, 2].set_ylim(-100, 5)
axs[0, 2].set_xlim(0, 1)
axs[0, 2].set_xticks([0, 1], [0, 1], fontsize=TICK_FS)
axs[0, 2].set_yticks([-75,0,50], [-75,0,50], fontsize=TICK_FS)
axs[0, 2].set_ylabel(r'$Re[\lambda_{attractor}]$', fontsize=LABEL_FS, labelpad=-5)

# -----------------------------------------------------------------------------
# Shared styling and output
# -----------------------------------------------------------------------------
for j in range(3):
    for k in range(2):
        axs[k, j].spines['top'].set_visible(False)
        axs[k, j].spines['right'].set_visible(False)
        axs[k, j].spines['left'].set_linewidth(2)
        axs[k, j].spines['bottom'].set_linewidth(2)
        #axs[k, j].tick_params(labelsize=TICK_FS)

plt.subplots_adjust(hspace=.5, bottom=.09, top=.91, left=.05, wspace=.32, right=.97)

plt.savefig('Fig5.pdf', dpi=100)
plt.show()


# Inactive numerical-separatrix diagnostic retained unchanged
"""
if i == 0:
    w0 = np.array([float(wEEval_rIzero), float(wEIval_rEzero)], dtype=float)
    J = jacobian_F(w0)

    # directions (dense fan)
    N_DIRS = 16
    angles = np.linspace(0.0, 2.0 * np.pi, N_DIRS, endpoint=False)
    dirs = [np.array([np.cos(th), np.sin(th)], dtype=float) for th in angles]

    EPS_SEED = 1e-5
    STEP_LEN = 1e-4
    N_STEPS  = 20000

    _labeled_sep = False
    for v in dirs:
        seed = w0 + EPS_SEED * v
        for time_dir in (+1, -1):
            br = rk4_streamline_unbounded(seed, step_len=STEP_LEN, n_steps=N_STEPS, direction=time_dir)
            if br.size:
                axs[1, i].plot(br[:, 0], br[:, 1], color='r', lw=1.25, alpha=0.95,
                               linestyle=':', label=("numerical separatrix" if not _labeled_sep else None))
                _labeled_sep = True
"""

