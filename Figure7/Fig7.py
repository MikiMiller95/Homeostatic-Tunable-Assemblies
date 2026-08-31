"""Create Figure 7 from the precomputed simulation, theory, and raster data.

The plotting calculations, data selection, plot calls, axes, labels, limits,
and plot order are unchanged from the uploaded Fig8.py file. The requested
output filename is changed from Fig8.pdf to Fig7.pdf.
"""

############################################################################
# Imports
############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.ticker as mticker

TITLE_FS  = 11   # titles
LABEL_FS  = 11   # x/y labels
TICK_FS   = 9   # ticks (matches your ft-1 usage later)
LEGEND_FS = 10   # legends

plt.rcParams.update({"axes.titlesize": TITLE_FS, "axes.labelsize": LABEL_FS, "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS, "legend.fontsize": LEGEND_FS})

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "text.latex.preamble": r"\usepackage{amsmath}"})

############################################################################
# Global style parameters
############################################################################
tick_ft  = 8      # xtick / ytick label size
lbl_ft   = 10     # axis-label size
title_ft = 10      # axes-title size
lt       = 1      # default line width for curves

############################################################################
# Simulation / figure parameters
############################################################################
# ---- sweep variable ------------------------------------------------------
c_xs          = [0., .2, .4 , .6, .8, 1.]        # values to iterate over
sigma   = 0.50                            # keep σ constant here
# -------------------------------------------------------------------------

N             = 500
ms_per_sec    = 1000.
scale_factor  = np.sqrt(N)
tau_rprim     = 2.  / ms_per_sec
w_EE_0        = 18. / scale_factor * tau_rprim
w_EI_0        = 18. / scale_factor * tau_rprim
w_IE          = 10. / scale_factor * tau_rprim
w_II          = 10. / scale_factor * tau_rprim
w_EX          = 20. / scale_factor * tau_rprim
w_IX          = 20. / scale_factor * tau_rprim
tau_wee       = 450
tau_wei       = 150.
tau_STDP      = 120 / 1000.
tau_r         = 8.  / 1000
tau_ou        = 8.   / 1000
T             = 110.
dt            = 0.1 / ms_per_sec
time          = np.linspace(0, T, int(T / dt))
loc           = 'Fig7'
skip          = 2000
time_reduced  = time[::skip]
seeds         = 10
aE, aI, b     = 20, 20, 20.

############################################################################
# Figure set-up (2 × 2 grid)
############################################################################
fig, ax = plt.subplots(2, 3,figsize=(7.5, 4.5), constrained_layout=False)

############################################################################
# 1) Weight-space line attractor (ax[0,0]) – identical to σ-script
############################################################################
ax_line = ax[1, 0]
ax_weight = ax[1,1]
ax_drift = ax[1,2]
ax_raster=ax[0,2]

max_wi         = 35
w_EE_values    = np.linspace(0., 35, 300) / scale_factor * tau_rprim
w_EI_values    = np.linspace(0., max_wi, 300) / scale_factor * tau_rprim
w_ee, w_ei     = np.meshgrid(w_EE_values, w_EI_values)

# W-matrix grid
W = np.zeros((len(w_EE_values), len(w_EI_values), 2, 2))
for iE, valEE in enumerate(w_EE_values):
    for iI, valEI in enumerate(w_EI_values):
        W[iE, iI, 0, 0] =  valEE
        W[iE, iI, 0, 1] = -valEI
        W[iE, iI, 1, 0] =  w_IE
        W[iE, iI, 1, 1] = -w_II

# Fixed-point rates
Wx   = np.array([[w_EX, 0], [0, w_IX]])
rx   = np.array([aE, aI])
fixed_points = np.zeros((2, len(w_EE_values), len(w_EI_values)))

for iE in range(len(w_EE_values)):
    for iI in range(len(w_EI_values)):
        M   = np.eye(2) - N * W[iE, iI, :, :]
        rhs = Wx @ rx * N
        fixed_points[:, iE, iI] = np.linalg.inv(M) @ rhs

re_fixed = fixed_points[0]
ri_fixed = fixed_points[1]
dwee     = re_fixed * (re_fixed - b) * tau_STDP / tau_wee
dwei     = ri_fixed * (re_fixed - b) * tau_STDP / tau_wei

# rI = 0 and rE = 0 lines
wEEval_rIzero = (aE*N*w_EX*w_IE + aI*w_IX) / (aI*N*w_IX)
wEIval_rEzero = (aE*w_EX*(1 + N*w_II))      / (aI*N*w_IX)
ax_line.axvline(wEEval_rIzero, color='k', linewidth=lt,linestyle=':')
ax_line.axhline(wEIval_rEzero, color='k', linewidth=lt,linestyle=':')

# Line attractor
wEE_range = np.linspace(-1, w_EE_values[-1], 200)
num       = (-b + b*N*wEE_range + aE*N*w_EX) * (1 + N*w_II)
den       = (b*N**2*w_IE + aI*N**2*w_IX)
wEI_line  = num / den
ax_line.plot(wEE_range, wEI_line, color='b', linewidth=lt/2)

# Oscillatory boundary
nonosc_stability = (-1 + N*w_EE_values - N*w_II + N**2*w_EE_values*w_II) / (N**2*w_IE)
ax_line.plot(w_EE_values, nonosc_stability, color='k', ls='--', lw=lt)

# Formatting
ax_line.set_xlim(0, w_EE_values[-1])
ax_line.set_ylim(0, w_EI_values[-1])
ax_line.set_xlabel(r'$w_{EE}$', fontsize=lbl_ft,labelpad=-10)
ax_line.set_ylabel(r'$w_{EI}$', fontsize=lbl_ft,labelpad=-10)
ax_line.set_title('Weight Space', fontsize=title_ft)

ax_line.set_xticks([0,.001], ['0','.001'], fontsize=tick_ft)
ax_line.set_yticks([0,.002], ['0','.002'], fontsize=tick_ft)
ax_line.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_line.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
ax_line.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)

ax_line.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_line.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
ax_line.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
ax_line.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
ax_line.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
# Place the rE=0 label using axes-x and data-y coordinates.
ax_line.text(.99, wEIval_rEzero-1e-4, r'$r_E^* = 0$', transform=ax_line.get_yaxis_transform(), va='center', ha='left', fontsize=lbl_ft)
############################################################################
# 2–4) Sweep over c_x (σ fixed)
############################################################################
cmap          = plt.cm.viridis
norm          = mcolors.Normalize(vmin=0, vmax=len(c_xs))


sim_drift     = []; sim_drift_std  = []
th_drift      = []; th_drift_std   = []

end_t = 70. 
start_t = 25
start_t, end_t = 25., 70.
idx_s  = np.argmin(np.abs(time_reduced - start_t))
idx_e  = np.argmin(np.abs(time_reduced - end_t))

time_theory = np.linspace(0,T,int(T/dt))
print('shape of time thoery',np.shape(time_theory))
idx_st  = np.argmin(np.abs(time_theory - start_t))
idx_et  = np.argmin(np.abs(time_theory - end_t))
for i, c_val in enumerate(c_xs):
    color = cmap(norm(i))

    seed_sim_drift     = [] #; sim_drift_std  = []
    seed_th_drift      = [] #; th_drift_std   = []

    # Per-seed storage
    mean_wee_seeds = np.zeros((seeds, len(time_reduced)))
    mean_theory_seeds = np.zeros((seeds, len(time)-1))
    mean_wei_seeds = np.zeros_like(mean_wee_seeds)


    for s in range(seeds):
        # ---------- theory trajectory ----------------------------------------
        # ---------- simulation CSVs ---------------------------------------
        df_wee = pd.read_csv(f'{loc}data/mean_ee_weights_Ne_{N}_T{T}_wee{w_EE_0}_wei{w_EI_0}_' f'wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_val}_' f'taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_' f'tauwee{tau_wee}_tauwei{tau_wei}_seed{s}.csv')
        mean_wee_seeds[s] = df_wee.to_numpy()[:-1, 0]

        df_wei = pd.read_csv(f'{loc}data/mean_ei_weights_Ne_{N}_T{T}_wee{w_EE_0}_wei{w_EI_0}_' f'wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_val}_' f'taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_' f'tauwee{tau_wee}_tauwei{tau_wei}_seed{s}.csv')
        mean_wei_seeds[s] = df_wei.to_numpy()[:-1, 0]

        df_theory_wee = pd.read_csv(f'{loc}data/W_theory_Ne_{N}_T{T}_wee{w_EE_0}_wei{w_EI_0}_' f'wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_val}_taur{tau_r}_tauSTDP{tau_STDP}_' f'tauou{tau_ou}_tauwee{float(tau_wee)}_tauwei{tau_wei}_seed{s}.csv')
        print(s,sigma, 'SEEDSnp.shape(df_theory_wee)',np.shape(df_theory_wee))
        mean_theory_seeds[s, :] = df_theory_wee.to_numpy()[:-1, 0]

        # --------- Drift integral 20–70 s (for panel ax[1,1]) ---------------


        diff_sim = np.diff(mean_wee_seeds[s,idx_s:idx_e])
        diff_th = np.diff(mean_theory_seeds[s,idx_st:idx_et])
        dt_sim = np.diff(time_reduced[idx_s:idx_e])
        dt_th = np.diff(time_theory[idx_st:idx_et])

        summed_sim = np.sum(diff_sim/dt_sim)*(time_reduced[1]-time_reduced[0])
        summed_th = np.sum(diff_th/dt_th)*(time_theory[1]-time_theory[0])
        seed_sim_drift.append(summed_sim)
        seed_th_drift.append(summed_th)


    # Aggregate seed statistics
    mean_wei = mean_wei_seeds.mean(axis=0)
    std_wei = mean_wei_seeds.std(axis=0)

    mean_wee = mean_wee_seeds.mean(axis=0)
    std_wee  = mean_wee_seeds.std(axis=0)

    mean_theory_wee = mean_theory_seeds.mean(axis=0)
    std_theory_wee  = mean_theory_seeds.std(axis=0)

    data = [mean_wee,mean_wei]
    for i in range(len(data)):
        dat = data[i]
        idx= np.where(np.abs(dat)>2)
        dat[idx]=2
        idx= np.where(np.abs(dat)>.1)
        std_wee[idx]=.00001
    data = [mean_theory_wee]
    for i in range(len(data)):
        dat = data[i]
        idx= np.where(np.abs(dat)>2)
        dat[idx]=2
        idx= np.where(np.abs(dat)>.1)
        std_theory_wee[idx]=.00001

    # --------- Plot wEE(t) for chosen c_x values (ax[0,1]) ---------------
    if c_val in [0., .4, 1.0]:
        ax_weight.plot(time_reduced, mean_wee, lw=lt, ls=':', color=color)
        ax_weight.fill_between(time_reduced, mean_wee-std_wee, mean_wee+std_wee, color=color, alpha=0.25)
        ax_weight.plot(time_theory[:-1], mean_theory_wee, color=color, linewidth=lt, linestyle='solid')
        ax_weight.fill_between(time_theory[:-1], mean_theory_wee - std_theory_wee, mean_theory_wee + std_theory_wee, color='k', alpha=0.3)
        if c_val==0.:
            ax_weight.plot(time_reduced, mean_wee, lw=lt, ls=':', color=color,label='Sim')
            ax_weight.plot(time_theory[:-1], mean_theory_wee, color=color,linewidth=lt, linestyle='solid',label='Theory')#,


    sim_drift.append(np.mean(seed_sim_drift))
    th_drift.append(np.mean(seed_th_drift))

    sim_drift_std.append(np.std(seed_sim_drift))
    th_drift_std.append(np.std(seed_th_drift))

    # --------- Weight-space traces for 2 exemplar c_x --------------------
    if c_val in [0., 1.]:
        ax_line.plot(mean_wee, np.abs(mean_wei), lw=lt+.5, color=( 'tab:purple' if c_val==0. else 'tab:green'), zorder=(20 if c_val==1. else 15),label=rf'$c_x$={c_val}')

ax_line.legend(fontsize=tick_ft, loc='center left',frameon=False)
ax_weight.legend(fontsize=tick_ft, loc='upper left',frameon=False)

cxs=[0.,.4,1.]
annotations = [fr"$c_x = {s}$" for s in cxs]
x_pos = 0.05   # adjust x position (fraction of axis width)
y_positions = [0.6, 0.5, 0.4]  # vertical placement in axis coords
cu_colors = [line.get_color() for line in ax_weight.get_lines()]
curve_colors=[cu_colors[0],cu_colors[4],cu_colors[6]]
print('curve_colors',curve_colors)
for text, y, color in zip(annotations, y_positions, curve_colors):
    ax_weight.text(x_pos, y,text,color=color,transform=ax_weight.transAxes,fontsize=tick_ft,ha='left', va='center')
ax_weight.set_xticks([5,100],[5,100])


# --------------------------------------------------------------------------
# PLOT RASTeR
# --------------------------------------------------------------------------
print('LOADING SPIKES')
print('================================================')
print('================================================')
lt = 1

# Simulation / figure parameters
N_E = N_I = 1000 #1000
ms_per_sec = 1000
T = 7.5
dt = 0.1 / ms_per_sec
time = np.linspace(0, T, int(T / dt))

loc = 'Driftdata/'
sigmas = [4.]
seeds = 530
b = 20.
tau_rprim = 2. / ms_per_sec
tau_r = 8. / ms_per_sec
tau_ou = 8. / ms_per_sec
tau_STDP = 120. / ms_per_sec
tau_wee = 225000
tau_wei = 75000.
c_x = 0.
wee_init = 10. / np.sqrt(N_E) * tau_rprim
wei_init = 10. / np.sqrt(N_E) * tau_rprim
wie = 10. / np.sqrt(N_E) * tau_rprim
wii = 10. / np.sqrt(N_E) * tau_rprim
start=6.78
end = 6.85

# Raster plots in ax[0,2]
for idx, sigma in enumerate(sigmas):
    fileX = f"{loc}spike_trains_X_Ne_{N_E}_T{T}_wee{wee_init}_wei{wei_init}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seeds}.csv"
    fileE = f"{loc}spike_trains_E_Ne_{N_E}_T{T}_wee{wee_init}_wei{wei_init}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seeds}.csv"
    fileI = f"{loc}spike_trains_I_Ne_{N_E}_T{T}_wee{wee_init}_wei{wei_init}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seeds}.csv"
    dfX = pd.read_csv(fileX)
    dfE = pd.read_csv(fileE)
    dfI = pd.read_csv(fileI)

    # ---- raster (identical time window as Fig-4) ------------------------------
    tmin, tmax = 6.75, 6.85
    tmin, tmax = 6.6, 7.3

    N_X = dfX.shape[0]
    Eoff, Ioff, Xoff = 0, N_E, N_E + N_I

    half = 10.  # half the tick height in row units
    for i in range(N_E):
        st = dfE.iloc[i].values; st = st[st > 0]
        st = st[np.isfinite(st)]
        st = st[(st >= tmin) & (st <= tmax)]
        y = Eoff + i
        ax_raster.vlines(st, y - half, y + half, colors='tab:red', linewidth=0.3)
    for i in range(N_I):
        st = dfI.iloc[i].values; st = st[st > 0]
        st = st[np.isfinite(st)]
        st = st[(st >= tmin) & (st <= tmax)]
        y = Ioff + i
        ax_raster.vlines(st, y - half, y + half, colors='tab:blue', linewidth=0.3)
    for i in range(N_X):
        st = dfX.iloc[i].values; st = st[st > 0]
        st = st[np.isfinite(st)]
        st = st[(st >= tmin) & (st <= tmax)]
        y = Xoff + i
        ax_raster.vlines(st, y - half, y + half, colors='k', linewidth=0.3)

c_x=1
ax_raster.set_title(rf"Asynch Raster ($c_x =$ ${c_x}$)", fontsize=title_ft)
ax_raster.axvline(6.8,color='k',linewidth=lt-.5,label='Corr on')
ax_raster.set_xticks([start, end],[start, end],fontsize=lbl_ft)
ax_raster.set_xticks([6.73,6.83],[-int(np.around((6.83-6.8)*1000,0)),int(np.around((6.83-6.8)*1000,0))],fontsize=lbl_ft)
#ax_raster.set_xlim(6.75, 6.85)
#ax_raster.set_xlim(tmin, tmax)
ax_raster.set_xlim(start-.03,end-.03)
ax_raster.set_yticks([0,2000],[0,2000],fontsize=lbl_ft)
ax_raster.set_xlabel("Time (ms)", fontsize=title_ft,labelpad=-5)
ax_raster.set_ylabel("Neuron Idx", fontsize=lbl_ft,labelpad=-15)
ax_raster.yaxis.labelpad = -2
ax_raster.legend(frameon=False,fontsize=lbl_ft,loc=(.5,.9))


############################################################################
# Panel formatting
############################################################################
# Mean wEE(t)
ax_weight.set_xlim(3., 110)
ax_weight.set_xlabel('Time (s)', fontsize=lbl_ft,labelpad=-10)
ax_weight.set_ylabel(r'$\frac{\bar{w}_{EE}}{w_0}$', fontsize=lbl_ft,labelpad=-10)
ax_weight.set_title(r'$\frac{\bar{w}_{EE}}{\bar{w}_0}$ over Time', fontsize=title_ft)

minval=-5e-6
maxval=3e-5
ax_weight.set_ylim(0.0015,.003)
# Integrated drift vs c_x
c_arr   = np.asarray(c_xs)
sim_drift      = np.asarray(sim_drift)
sim_drift_std  = np.asarray(sim_drift_std)
th_drift       = np.asarray(th_drift)
th_drift_std   = np.asarray(th_drift_std)

ax_drift.plot(c_arr, th_drift/w_EE_0, color='k', label='Theory')
ax_drift.fill_between(c_arr, (th_drift - th_drift_std/np.sqrt(seeds))/w_EE_0, (th_drift + th_drift_std/np.sqrt(seeds))/w_EE_0,color='k', alpha=.2)
ax_drift.plot(c_arr, sim_drift/w_EE_0, color='tab:blue', label='Sim')
ax_drift.fill_between(c_arr, (sim_drift - sim_drift_std/np.sqrt(seeds))/w_EE_0, (sim_drift + sim_drift_std/np.sqrt(seeds))/w_EE_0, color='tab:blue', alpha=.2)
ax_drift.set_xlabel(r'$c$ (EI Input Correlation)', fontsize=lbl_ft,labelpad=-10)
ax_drift.set_ylabel(r'$\frac{\Delta \bar{w}_{EE}}{w_0}$', fontsize=lbl_ft,labelpad=-15)
ax_drift.set_title('Weight Change vs $c$', fontsize=title_ft)
ax_drift.legend(fontsize=tick_ft, loc='upper right',frameon=False)
ax_drift.set_yticks([0,.25],[0,.25])
ax_drift.set_xticks([0,1],[0,1])
w_ticks=[1.,2.]
scaled_w_ticks=np.around(np.array(w_ticks)*w_EE_0,2)
ax_weight.set_yticks(np.array([1,2])*w_EE_0,[1,2])

for ax_ in ax.flat:
    ax_.tick_params(labelsize=tick_ft)
    for spine in ['top', 'right']:
        ax_.spines[spine].set_visible(False)
    ax_.spines['bottom'].set_linewidth(lt)
    ax_.spines['left'].set_linewidth(lt)

ax[0,0].tick_params(labelsize=tick_ft)
axs_list=[ax[0,0],ax[0,1]]
for ax in axs_list:
    for spine in ['top', 'right','left','bottom']:
        ax.spines[spine].set_visible(False)
        ax.set_yticks([],[])
        ax.set_xticks([],[])

TITLE_FS  = 11   # titles
LABEL_FS  = 11   # x/y labels
TICK_FS   = 9   # ticks (matches your ft-1 usage later)
LEGEND_FS = 10   # legends

plt.rcParams.update({"axes.titlesize": TITLE_FS, "axes.labelsize": LABEL_FS, "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS, "legend.fontsize": LEGEND_FS})

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "text.latex.preamble": r"\usepackage{amsmath}"})


plt.subplots_adjust(left=0.05, right=0.97, bottom=0.08, top=0.93, hspace=0.4, wspace=0.25)

############################################################################
# Save / show
############################################################################
plt.savefig('Fig7.pdf', dpi=150)
plt.show()

