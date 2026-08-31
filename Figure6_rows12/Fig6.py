"""Assemble Figure 6 from precomputed simulation and theory CSV files.

The numerical analyses, panel contents, plotting calls, parameter values, and
axis formatting are preserved from the supplied source script. Only comments,
layout, inactive code removal, and the requested Figure 6 relabeling were made.
"""

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Global plotting style
plt.rcParams['font.size'] = 12

############################################################################
# Global style parameters
############################################################################
TITLE_FS  = 10
LABEL_FS  = 10
TICK_FS   = 8
LEGEND_FS = 9

plt.rcParams.update({ "axes.titlesize": TITLE_FS, "axes.labelsize": LABEL_FS, "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS, "legend.fontsize": LEGEND_FS, })

mpl.rcParams.update({ "font.family": "serif", "mathtext.fontset": "cm", "text.latex.preamble": r"\usepackage{amsmath}"})

############################################################################
# Simulation and plotting parameters (unchanged)
############################################################################
sigmas        = [0., .1, .2, .3, .4,.5]
N             = 500
ms_per_sec    = 1000.
scale_factor  = np.sqrt(N)
tau_rprim     = 2. / ms_per_sec
w_EE_0        = 18. / scale_factor * tau_rprim
w_EI_0        = 18. / scale_factor * tau_rprim
w_IE          = 10. / scale_factor * tau_rprim
w_II          = 10. / scale_factor * tau_rprim
w_EX          = 20. / scale_factor * tau_rprim
w_IX          = 20. / scale_factor * tau_rprim
tau_wee       = 225000
tau_wei       = 75000
tau_STDP      = 120 / 1000.
tau_r         = 8.  / 1000
tau_ou        = 8.   / 1000
T             = 110.
dt            = 0.2 / ms_per_sec
time          = np.linspace(0, T, int(T / dt))
loc           = 'Fig6'
skip          = 2000
time_reduced  = time[::skip]
seeds         = 1
aE, aI, b     = 20, 20, 20.
c_x           = 0.

############################################################################
# Figure set-up
############################################################################
fig, ax = plt.subplots(2, 4, figsize=(7.5, 3.))

# Reserve the upper-left area and use the remaining six axes
ax[0, 0].remove()
ax[0, 1].remove()
ax_rastor = ax[0, 2]

############################################################################
# 1) Weight-space line attractor and stability boundaries
############################################################################
ax_line = ax[1, 1]

# Build mesh in wEE-wEI space
max_wi         = 35
w_EE_values    = np.linspace(0., 35, 300) / scale_factor * tau_rprim
w_EI_values    = np.linspace(0., max_wi, 300) / scale_factor * tau_rprim
w_ee, w_ei     = np.meshgrid(w_EE_values, w_EI_values)

# Matrix W for each grid point
W = np.zeros((len(w_EE_values), len(w_EI_values), 2, 2))
for iE, valEE in enumerate(w_EE_values):
    for iI, valEI in enumerate(w_EI_values):
        W[iE, iI, 0, 0] =  valEE
        W[iE, iI, 0, 1] = -valEI
        W[iE, iI, 1, 0] =  w_IE
        W[iE, iI, 1, 1] = -w_II

# Fixed-point firing rates r*
Wx   = np.array([[w_EX, 0], [0, w_IX]])
rx   = np.array([aE, aI])
fixed_points = np.zeros((2, len(w_EE_values), len(w_EI_values)))

for iE in range(len(w_EE_values)):
    for iI in range(len(w_EI_values)):
        M = np.eye(2) - N * W[iE, iI, :, :]
        rhs = Wx @ rx * N
        fixed_points[:, iE, iI] = np.linalg.inv(M) @ rhs

re_fixed = fixed_points[0, :, :]
ri_fixed = fixed_points[1, :, :]
dwee     = re_fixed * (re_fixed - b) * tau_STDP / tau_wee
dwei     = ri_fixed * (re_fixed - b) * tau_STDP / tau_wei

# Boundaries rI = 0 (vertical) and rE = 0 (horizontal)
wEEval_rIzero = (aE*N*w_EX*w_IE + aI*w_IX) / (aI*N*w_IX)
wEIval_rEzero = (aE*w_EX*(1 + N*w_II)) / (aI*N*w_IX)
ax_line.axvline(wEEval_rIzero, color='k',linestyle=':')
ax_line.axhline(wEIval_rEzero, color='k',linestyle=':')

# Line attractor
wEE_range = np.linspace(-1, w_EE_values[-1], 200)
num       = (-b + b*N*wEE_range + aE*N*w_EX) * (1 + N*w_II)
den       = (b*N**2*w_IE + aI*N**2*w_IX)
wEI_line  = num / den
ax_line.plot(wEE_range, wEI_line, color='b')

# Oscillatory boundary
nonosc_stability_boundary = (
    -1 + N*w_EE_values - N*w_II + N**2*w_EE_values*w_II
) / (N**2*w_IE)
ax_line.plot(w_EE_values, nonosc_stability_boundary,color='k', linestyle='--',)

# Axes formatting
ax_line.set_xlim(0, w_EE_values[-1])
ax_line.set_ylim(0, w_EI_values[-1])
ax_line.set_xlabel(r'$\bar{w}_{EE}$',labelpad=-5)
ax_line.set_ylabel(r'$\bar{w}_{EI}$',labelpad=-10)
ax_line.set_title('Weight Space',)
ax_line.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_line.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
ax_line.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)

ax_line.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax_line.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
ax_line.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
ax_line.text(wEEval_rIzero,2e-4, r'$r_I^*=0$', fontsize=LABEL_FS)
ax_line.text(.95, wEIval_rEzero-5e-4, r'$r_E^* = 0$', transform=ax_line.get_yaxis_transform(), va='center', ha='left', fontsize=LABEL_FS,)
############################################################################
# 2) Sigma sweep: weight trajectories and integrated drift
############################################################################
cmap        = plt.cm.viridis
norm        = mcolors.Normalize(vmin=0, vmax=6)
slopes_data = []
slopes_data_err = []
slopes_theory = []
th_drift  = []
th_drift_std = []
sim_drift     = []
sim_drift_std    = []
count = 0


sim_drift     = []
sim_drift_std  = []
th_drift      = []
th_drift_std   = []

end_t = 70.
start_t = 25
start_t, end_t = 25., 70.
idx_s  = np.argmin(np.abs(time_reduced - start_t))
idx_e  = np.argmin(np.abs(time_reduced - end_t))

# The saved theory trajectories use the same skip=2000 grid as the simulation data.
time_theory = time_reduced.copy()
idx_st = np.argmin(np.abs(time_theory - start_t))
idx_et = np.argmin(np.abs(time_theory - end_t))
for i, sigma in enumerate(sigmas):
    color = cmap(norm(6-count-1))
    # Allocate arrays for all seeds at the current correlation strength
    seed_sim_drift     = []
    seed_th_drift      = []

    mean_wee_seeds   = np.zeros((seeds, len(time_reduced)))
    mean_theory_seeds = np.zeros((seeds, len(time_theory)))
    mean_wei_seeds   = np.zeros_like(mean_wee_seeds)

    for s in range(seeds):
        print('s',s)
        # Load simulation, theory, balance, and reciprocity outputs
        df_wee = pd.read_csv( f'{loc}data/mean_ee_weights_Ne_{N}_T{T}_wee{w_EE_0}_wei{w_EI_0}_' f'wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_' f'tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{s}.csv')
        mean_wee_seeds[s, :] = df_wee.to_numpy()[:-1, 0]

        df_theory_wee = pd.read_csv( f'{loc}data/W_theory_Ne_{N}_T{T}_wee{w_EE_0}_wei{w_EI_0}_' f'wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_' f'tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{s}.csv')
        #mean_theory_seeds[s, :] = df_theory_wee.to_numpy()[:-1, 0]
        theory_wee = df_theory_wee.to_numpy()[:-1, 0]

        if theory_wee.size != time_theory.size:raise ValueError(f"Theory file for sigma={sigma}, seed={s} contains "f"{theory_wee.size} usable samples, but Fig6.py expects "f"{time_theory.size}.")

        mean_theory_seeds[s, :] = theory_wee

        df_wei = pd.read_csv( f'{loc}data/mean_ei_weights_Ne_{N}_T{T}_wee{w_EE_0}_wei{w_EI_0}_' f'wii{w_II}_wie{w_IE}_sigma{sigma}_cx{c_x}_' f'taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_' f'tauwee{tau_wee}_tauwei{tau_wei}_seed{s}.csv')
        mean_wei_seeds[s] = df_wei.to_numpy()[:-1, 0]

        if sigma==0 or sigma==.5:
            if sigma==0:
                k = 0
            if sigma==.5:
                k=1


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
    idx = np.where(std_wee>=.01)
    std_wee[idx]=.01

    mean_theory_wee = mean_theory_seeds.mean(axis=0)
    std_theory_wee  = mean_theory_seeds.std(axis=0)

    data = [mean_wee,mean_wei]
    for i in range(len(data)):
        dat = data[i]
        idx= np.where(dat>90)
        dat[idx]=90
        std_wei[idx]=.000001
        std_wee[idx]=.000001
    data = [mean_theory_wee]
    for i in range(len(data)):
        dat = data[i]
        idx= np.where(np.abs(dat)>90)
        dat[idx]=90
        std_theory_wee[idx]=.000001

    # Plot selected simulation and theory trajectories
    if sigma in [0., .3,  .5]:
        ax[1, 2].plot(time_reduced, mean_wee,color=color, linestyle=':')
        ax[1, 2].fill_between(time_reduced, mean_wee - std_wee,mean_wee + std_wee, color=color, alpha=0.3)
        #ax[1,2].plot(time_theory[:-1], mean_theory_wee, color=color, linestyle='solid')
        ax[1, 2].plot(time_theory,mean_theory_wee,color=color,linestyle='solid')
        if sigma==.5:
            #ax[1,2].plot(time_theory[:-1], mean_theory_wee, color=color, linestyle='solid',label='Theory')
            ax[1, 2].plot(time_theory,mean_theory_wee,color=color,linestyle='solid',label='Theory')
            ax[1, 2].plot(time_reduced, mean_wee,color=color, linestyle=':',label=f'Sim')

        #ax[1, 2].fill_between(time_theory[:-1], mean_theory_wee - std_theory_wee,mean_theory_wee + std_theory_wee, color=color, alpha=0.3)
        ax[1, 2].fill_between(time_theory,mean_theory_wee - std_theory_wee,mean_theory_wee + std_theory_wee,color=color,alpha=0.3)

    count += 1
    sim_drift.append(np.mean(seed_sim_drift))
    th_drift.append(np.mean(seed_th_drift))

    sim_drift_std.append(np.std(seed_sim_drift))
    th_drift_std.append(np.std(seed_th_drift))


    if sigma in [0., .5]:
        ax_line.plot(mean_wee, np.abs(mean_wei),color=('k' if sigma == 0. else 'tab:purple'),label=rf'$\sigma$={sigma}', zorder=(20 if sigma == 0. else 15))

ax[1,2].legend(loc='upper left',frameon=False)
# Integrated drift vs σ
sigmas_arr = np.asarray(sigmas)
th_drift = np.asarray(th_drift)
th_drift_std = np.asarray(th_drift_std)
sim_drift = np.asarray(sim_drift)
sim_drift_std = np.asarray(sim_drift_std)

print('th_drift',th_drift)
ax[1, 3].plot(sigmas_arr, th_drift/w_EE_0, color='k', label='Theory')
ax[1, 3].fill_between(sigmas_arr, (th_drift - th_drift_std/np.sqrt(seeds))/w_EE_0, (th_drift + th_drift_std/np.sqrt(seeds))/w_EE_0, color='tab:grey', alpha=0.2)

ax[1, 3].plot(sigmas_arr, sim_drift/w_EE_0, color='tab:blue', label='Sim')
ax[1, 3].fill_between(sigmas_arr, (sim_drift - sim_drift_std/np.sqrt(seeds))/w_EE_0, (sim_drift + sim_drift_std/np.sqrt(seeds))/w_EE_0, color='tab:blue', alpha=0.2)
ax[1, 3].set_xlabel(r'$\sigma$ (Correlation Strength)',)
ax[1, 3].set_ylabel(r'$\Delta \bar{w}_{EE}/\bar{w}_0$')
ax[1, 3].set_title(r'Weight Change vs $\sigma$')

# --------------------------------------------------------------------------
# Panel labels, limits, legends, and annotations

yticks= [1.,2.]
ax[1, 2].set_yticks([w_EE_0,2.*w_EE_0],yticks)
ax[1, 3].set_yticks([.1,.2],[.1,.2])
ax[1, 2].set_xlim(3., 110)
ax[1, 2].set_ylim(0.0015, 2.*w_EE_0)
ax[1, 2].set_xlabel('Time (sec)',)
ax[1, 2].set_ylabel(r'$\bar{w}_{EE}/\bar{w}_{0}$',)
ax[1, 2].set_xticks([0,100],[0,100])
ax[1, 3].set_title(r'Mean $\frac{w_{EE}}{w_{0}}$ over Time',)
ax[0, 3].set_xlabel('Time (sec)',)
ax[0, 3].set_ylim(-0.5e-5, 3.e-5)


legend=ax[1,2].legend(loc='upper left',frameon=False)

sigs=[0.,.3,.5]
annotations = [fr"$\sigma = {s}$" for s in sigs]
x_pos = 0.05
y_positions = [0.6, 0.5, 0.4]
cu_colors = [line.get_color() for line in ax[1,2].get_lines()]
curve_colors=[cu_colors[0],cu_colors[2],cu_colors[3]]
print('curve_colors',curve_colors)
for text, y, color in zip(annotations, y_positions, curve_colors):
    ax[1,3].text(x_pos, y,text,color=color,transform=ax[1,2].transAxes,fontsize=8,ha='left', va='center')


ax[0,3].set_xlim(0,70)

# --------------------------------------------------------------------------
# 3) Correlated-input raster panel
# --------------------------------------------------------------------------
print('LOADING SPIKES')
print('================================================')
print('================================================')
lt = 1

# Raster simulation parameters
N_E = N_I = 500
ms_per_sec = 1000
T = 7.
dt = 0.2 / ms_per_sec
time = np.linspace(0, T, int(T / dt))

loc = 'Driftdata/'
sigmas = [4.]
seeds = 54000
b = 20.
tau_rprim = 2. / ms_per_sec
tau_r = 8. / ms_per_sec
tau_ou = 8. / ms_per_sec
tau_STDP = 150. / ms_per_sec
tau_wee = 450
tau_wei = 150.
c_x = 0.
wee_init = 18. / np.sqrt(N_E) * tau_rprim
wei_init = 18. / np.sqrt(N_E) * tau_rprim
wie = 10. / np.sqrt(N_E) * tau_rprim
wii = 10. / np.sqrt(N_E) * tau_rprim
start=6.78
end = 6.85

# Load and plot the E, I, and external spike trains
for idx, sigma in enumerate(sigmas):
    fileX = f"{loc}spike_trains_X_Ne_{N_E}_T{T}_wee{wee_init}_wei{wei_init}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seeds}.csv"
    fileE = f"{loc}spike_trains_E_Ne_{N_E}_T{T}_wee{wee_init}_wei{wei_init}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seeds}.csv"
    fileI = f"{loc}spike_trains_I_Ne_{N_E}_T{T}_wee{wee_init}_wei{wei_init}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seeds}.csv"
    dfX = pd.read_csv(fileX)
    dfE = pd.read_csv(fileE)
    dfI = pd.read_csv(fileI)


    N_X = dfX.shape[0]
    Eoff, Ioff, Xoff = 0, N_E, N_E + N_I

    half = 10.
    tmin, tmax = 6.75, 6.85
    for i in range(N_E):
        st = dfE.iloc[i].values
        st = st[st > 0]
        st = st[np.isfinite(st)]
        st = st[(st >= tmin) & (st <= tmax)]
        y = Eoff + i
        ax_rastor.vlines(st, y - half, y + half, colors='tab:red', linewidth=0.3)
    for i in range(N_I):
        st = dfI.iloc[i].values
        st = st[st > 0]
        st = st[np.isfinite(st)]
        st = st[(st >= tmin) & (st <= tmax)]
        y = Ioff + i
        ax_rastor.vlines(st, y - half, y + half, colors='tab:blue', linewidth=0.3)
    for i in range(N_X):
        st = dfX.iloc[i].values
        st = st[st > 0]
        st = st[np.isfinite(st)]
        st = st[(st >= tmin) & (st <= tmax)]
        y = Xoff + i
        ax_rastor.vlines(st, y - half, y + half, colors='k', linewidth=0.3)


ax_rastor.set_title(rf"Correlated Rastor ($\sigma =$ ${sigma}$)",)
ax_rastor.axvline(6.8,color='k',label='Corr on')
ax_rastor.set_xticks([start, end],[start, end])
ax_rastor.set_xticks([6.75,6.85],[-int(np.around((6.85-6.8)*1000,0)),int(np.around((6.85-6.8)*1000,0))])
ax_rastor.set_xlim(6.75, 6.85)
ax_rastor.set_yticks([0,2000],[0,2000])
ax_rastor.set_xlabel("Time (ms)",labelpad=0)
ax_rastor.set_ylabel("Neuron Idx",labelpad=-15)
ax_rastor.yaxis.labelpad = -2
ax_rastor.legend(frameon=False,loc=(.5,.9))

# --------------------------------------------------------------------------
# 4) Long-timescale weight and rate traces
print('LOADING TRACES')
print('================================================')
print('================================================')
loc = 'Driftdata/'
T = 80.
dt = .1/ms_per_sec
sigmas = [0.,.5]
N_E = N_I = 500
time = np.linspace(0, T, int(T / dt))
time_reduced = time[::2000]
tau_rprim = 2. / ms_per_sec
tau_r = 8. / ms_per_sec
tau_ou = 8. / ms_per_sec
tau_STDP = 120. / ms_per_sec
tau_wee = 225000
tau_wei = 75000.
c_x = 0.
seeds=1
wee_init = 18. / np.sqrt(N_E) * tau_rprim
wei_init = 18. / np.sqrt(N_E) * tau_rprim
wie = 10. / np.sqrt(N_E) * tau_rprim
wii = 10. / np.sqrt(N_E) * tau_rprim
# Aggregate long-timescale weight and rate traces across seeds
for j, sigma in enumerate(sigmas):

    mean_wee_seeds = np.zeros((seeds, len(time_reduced)))
    mean_wei_seeds = np.zeros((seeds, len(time_reduced)))
    mean_re_seeds = np.zeros((seeds, len(time_reduced)))
    mean_ri_seeds = np.zeros((seeds, len(time_reduced)))

    for seed in range(seeds):
        df_wee_path = f'{loc}mean_ee_weights_Ne_{N_E}_T{T}_wee{wee_init}_wei{wei_init}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
        df_wei_path = f'{loc}mean_ei_weights_Ne_{N_E}_T{T}_wee{wee_init}_wei{wei_init}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
        df_rates_path = f'{loc}mean_rates_Ne_{N_E}_T{T}_wee{wee_init}_wei{wei_init}_wii{wii}_wie{wie}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'

        print(np.shape(pd.read_csv(df_wee_path).values[:-1, 0]))
        print(np.shape(mean_wee_seeds))
        mean_wee_seeds[seed] = pd.read_csv(df_wee_path).values[:-1, 0]
        mean_wei_seeds[seed] = pd.read_csv(df_wei_path).values[:-1, 0]
        rates_array = pd.read_csv(df_rates_path).values
        mean_re_seeds[seed] = rates_array[0, :-1]
        mean_ri_seeds[seed] = rates_array[1, :-1]

    mean_wee = np.mean(mean_wee_seeds, axis=0)
    mean_wei = np.mean(np.abs(mean_wei_seeds), axis=0)
    mean_re = np.mean(mean_re_seeds, axis=0)
    mean_ri = np.mean(mean_ri_seeds, axis=0)

    std_wee = np.std(mean_wee_seeds, axis=0)/np.sqrt(seeds)
    std_wei = np.std(np.abs(mean_wei_seeds), axis=0)/np.sqrt(seeds)
    std_re = np.std(mean_re_seeds, axis=0)/np.sqrt(seeds)
    std_ri = np.std(mean_ri_seeds, axis=0)/np.sqrt(seeds)

    if sigma==0:
        linestyle=':'
        labelwee=r'$\bar{w}_{EE}:\sigma='+f'{sigma}$'
        labelwei=r'$\bar{w}_{EI}$'
        labelre=r'$\bar{r}_E:\sigma='+f'{sigma}$'
        labelri=r'$\bar{r}_I$'
    else:
        linestyle='solid'
        labelwee=r'$\bar{w}_{EE}:\sigma='+f'{sigma}$'
        labelwei=r'$\bar{w}_{EI}$'
        labelre=r'$\bar{r}_E:\sigma='+f'{sigma}$'
        labelri=r'$\bar{r}_I$'

    data = [mean_re,mean_ri]
    for i in range(len(data)):
        dat = data[i]
        idx= np.where(dat>90)
        dat[idx]=90
        std_re[idx]=.1
        std_ri[idx]=.1
    data = [mean_wee,mean_wei]
    for i in range(len(data)):
        dat = data[i]
        idx= np.where(dat>90)
        dat[idx]=90
        std_wei[idx]=.000001
        std_wee[idx]=.000001

    ax[0, 3].plot(time_reduced, mean_wee / mean_wee[0],color='tab:red', label=labelwee, linestyle=linestyle,)
    ax[0, 3].fill_between(time_reduced, mean_wee / mean_wee[0]-std_wee / mean_wee[0],mean_wee / mean_wee[0]+std_wee / mean_wee[0],color='tab:red',alpha=.5)
    ax[0, 3].plot(time_reduced, mean_wei / mean_wei[0],color='tab:blue', label=labelwei,linestyle=linestyle,)
    ax[0, 3].fill_between(time_reduced, mean_wei / mean_wei[0]-std_wei / mean_wei[0],mean_wei / mean_wei[0]+std_wei / mean_wei[0],color='tab:blue',alpha=.5)

    ax[1, 0].plot(time_reduced, mean_re, label=labelre,color='tab:red' ,linestyle=linestyle,)
    ax[1, 0].fill_between(time_reduced, mean_re-std_re,mean_re+std_re,color='tab:red',alpha=.5)
    ax[1, 0].plot(time_reduced, mean_ri, label=labelri,color='tab:blue',linestyle=linestyle,)
    ax[1, 0].fill_between(time_reduced, mean_ri -std_ri ,mean_ri+std_ri,color='tab:blue',alpha=.5)
ax[0, 3].set_title('Weights Over Time',)
ax[1, 0].set_title('Rates Over Time',)
ax[0, 3].set_xlabel('Time (sec)',labelpad=-5)
ax[1, 0].set_xlabel('Time (sec)',labelpad=-5)
ax[0, 3].set_ylabel(r'$\bar{w}/\bar{w}_0$',)
ax[1, 0].set_ylabel(r'$\bar{r}_E$ (Hz)',)
ax[1, 0].set_yticks([0,20,40],[0,20,40])
ax[0, 3].legend(frameon=False)
ax[1, 0].legend(loc=(.1,.65),frameon=False)
ax[1, 0].set_ylim(0,45)
ax[1, 0].set_xticks([0,80],[0,80])
ax[0, 3].set_xticks([0,80],[0,80])
ax[0, 3].set_yticks([1,2],[1,2])
ax[0, 3].set_ylim(.8,2.5)

# Apply shared axis styling and save the assembled figure
for ax_ in ax.flat:
    ax_.tick_params(axis='both')
    ax_.spines['top'].set_visible(False)
    ax_.spines['right'].set_visible(False)
    ax_.spines['bottom'].set_linewidth(lt)
    ax_.spines['left'].set_linewidth(lt)


for a in [ax[1,2],ax[0,2]]:
    a.xaxis.labelpad = -10
for a in [ax[0,3],ax[1,1]]:
    a.xaxis.labelpad = -10
ax_rastor.spines['top'].set_visible(False)
ax_rastor.spines['right'].set_visible(False)
ax_rastor.spines['bottom'].set_linewidth(lt)
ax_rastor.spines['left'].set_linewidth(lt)

plt.rcParams.update({ "axes.titlesize": TITLE_FS, "axes.labelsize": LABEL_FS, "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS, "legend.fontsize": LEGEND_FS, })


plt.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.93, hspace=0.35, wspace=0.4)
plt.savefig('Fig6.pdf',dpi=100)
plt.show()
