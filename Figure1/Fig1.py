# Assemble Figure 1 from schematic calculations and pre-generated simulation data.

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global typography and line-width settings used throughout the figure.
ft = 9
lt = 1.0
SPINE_W = 1.0
plt.rcParams['font.size'] = ft
plt.rcParams['lines.linewidth'] = lt
mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'text.latex.preamble': '\\usepackage{amsmath}'})


# Simulate repeated pre/post spike pairings for the schematic STDP panels.
def simulate_stdp_full(freq, delta_t_sec, n_pairs=60, w_init=0.5, dt_sec=0.0001, eta=0.001, r0=2.0, tau_STDP_sec=0.02):
    """
    Simulate STDP pairing and return:
      - time array
      - yE trace (postsynaptic trace)
      - weight trace
      - presynaptic spike times
      - postsynaptic spike times
    """
    # Construct the pre- and postsynaptic spike times.
    interval_sec = 1.0 / freq
    t_pre = np.arange(n_pairs) * interval_sec
    t_post = t_pre + delta_t_sec

    # Shift both trains when a negative delay would place the first spike before zero.
    earliest = min(t_pre[0], t_post[0])
    if earliest < 0.0:
        t_pre -= earliest
        t_post -= earliest
    T_sec = max(t_pre[-1], t_post[-1]) + 5.0 * tau_STDP_sec

    # Merge the spike trains into one time-ordered event sequence.
    events = sorted([(t, 'pre_I') for t in t_pre] + [(t, 'post_E') for t in t_post], key=lambda x: x[0])

    # Initialize the synaptic weight and exponentially decaying spike traces.
    w = w_init
    yE = 0.0
    yI = 0.0
    alpha = 2.0 * r0 * tau_STDP_sec
    num_steps = int(np.ceil(T_sec / dt_sec))
    time = np.linspace(0, T_sec, num_steps)
    evt_idx = 0
    evt_time, evt_kind = events[evt_idx]
    track_yE = []
    track_w = []
    spike_times_pre = []
    spike_times_post = []

    # Decay the traces and apply the event-triggered weight updates.
    for t_now in time:
        decay = dt_sec / tau_STDP_sec
        yE -= yE * decay
        yI -= yI * decay
        while evt_idx < len(events) and np.isclose(evt_time, t_now, atol=dt_sec / 2):
            if evt_kind == 'pre_I':
                w += eta * (yE - alpha)
                yI += 1.0
                spike_times_pre.append(t_now)
            else:
                w += eta * yI
                yE += 1.0
                spike_times_post.append(t_now)
            evt_idx += 1
            if evt_idx < len(events):
                evt_time, evt_kind = events[evt_idx]
            else:
                evt_time = np.inf
        track_yE.append(yE)
        track_w.append(w)
    return (time, np.array(track_yE), np.array(track_w), np.array(spike_times_pre), np.array(spike_times_post))

# Parameters for the paired-spike examples shown in the central panels.
dt_sec = 0.0002
tau_STDP_sec = 0.5
freq = 20.0
eta = 1 / 300.0
r0 = 20.0
w_init = 50.0 / np.sqrt(500.0)

# Generate the two example pairing conditions.
time_left, yE_left, w_left, spk_pre_left, spk_post_left = simulate_stdp_full(freq=freq - 1.0, delta_t_sec=0.02, n_pairs=100, w_init=w_init, dt_sec=dt_sec, eta=eta, r0=r0, tau_STDP_sec=tau_STDP_sec)
time_right, yE_right, w_right, spk_pre_right, spk_post_right = simulate_stdp_full(freq=freq + 5, delta_t_sec=0.001, n_pairs=100, w_init=w_init, dt_sec=dt_sec, eta=eta, r0=r0, tau_STDP_sec=tau_STDP_sec)

# Create the main figure grid and reserve columns for the schematic and data panels.
gap_inner = -0.15
gap_outer = 1.25
fig = plt.figure(figsize=(7.5, 4.5))
gs = fig.add_gridspec(nrows=5, ncols=5, wspace=0.25, hspace=0.5, width_ratios=[1.2, 1, gap_inner, 1, gap_outer], height_ratios=[0.7, 0.7, 1.2, 1.2, 1.9])
ax_empty0 = fig.add_subplot(gs[0, 0])
ax_empty0.axis('off')
ax_empty1 = fig.add_subplot(gs[1, 0])
ax_empty1.axis('off')
ax_plastic = fig.add_subplot(gs[0:3, 0])
ax_empty4 = fig.add_subplot(gs[4, 0])
ax_empty4.axis('off')
axes_left = [fig.add_subplot(gs[r, 1]) for r in range(5)]
axes_right = [fig.add_subplot(gs[r, 3]) for r in range(5)]
axes_gap_inner = [fig.add_subplot(gs[r, 2]) for r in range(5)]
axes_gap_outer = [fig.add_subplot(gs[r, 4]) for r in range(5)]
for ax in axes_gap_inner + axes_gap_outer:
    ax.axis('off')

# Plot the left paired-spike example and its synaptic-weight trajectory.
axes_left[0].scatter(spk_pre_left, np.zeros_like(spk_pre_left), marker='|', s=1000, color='tab:green')
axes_left[1].scatter(spk_post_left, np.zeros_like(spk_post_left), marker='|', s=1000, color='tab:orange')
axes_left[4].plot(time_left, w_left, lw=lt, color='tab:red')
start1 = 0.625
end1 = 0.675
axes_left[4].set_ylim(1.71, 1.78)
axes_left[4].set_xlim(start1, end1)
axes_left[1].set_xlim(start1, end1)
axes_left[0].set_xlim(start1, end1)

# Plot the right paired-spike example and its synaptic-weight trajectory.
axes_right[0].scatter(spk_pre_right, np.zeros_like(spk_pre_right), marker='|', s=1000, color='tab:green')
axes_right[1].scatter(spk_post_right, np.zeros_like(spk_post_right), marker='|', s=1000, color='tab:orange')
axes_right[4].plot(time_right, w_right, lw=lt, color='tab:red')
start2 = 2.99
end2 = 2.99 + 0.02
axes_right[4].set_xlim(start2, end2)
axes_right[1].set_xlim(start2, end2)
axes_right[0].set_xlim(start2, end2)
axes_right[4].set_ylim(2.41, 2.49)

# Apply labels, tick placement, and shared axis styling to the paired examples.
axes_left[0].set_ylabel('Spike0', fontsize=ft)
axes_left[1].set_ylabel('Spike5', fontsize=ft)
axes_left[2].set_ylabel('$\\Lambda_{*}(t)$', fontsize=ft)
axes_left[3].set_ylabel('$\\Lambda_{*}(t)$', fontsize=ft)
axes_left[4].set_ylabel('$w _{ .*\\longrightarrow *. }$', fontsize=ft)
axes_left[4].set_xlabel('Time (ms)', fontsize=ft)
axes_right[4].set_xlabel('Time (ms)', fontsize=ft)
for col_axes in [axes_left, axes_right]:
    for j, ax in enumerate(col_axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if j != 4:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
for col_axes in [axes_left, axes_right]:
    for j, ax in enumerate(col_axes):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if j == 4:
            if col_axes is axes_left:
                xticks = np.array([start1, end1])
                labels = (xticks - start1) * 1000
            else:
                xticks = np.array([start2, end2])
                labels = (xticks - start2) * 1000
            ax.set_xticks(xticks)
            ax.set_xticklabels(labels.astype(int), fontsize=ft)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.tick_params(axis='x', which='both', length=0)
            ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])


# Compute and plot the schematic STDP curve.
freqs = [20]
colors = cm.viridis(np.linspace(0, 1, 5))
delta_ts = np.linspace(-40, 40, 150) / 1000.0
for i in range(len(freqs)):
    freq = freqs[i]
    dws = [simulate_stdp_full(freq=freq, delta_t_sec=dt, n_pairs=600, w_init=w_init, dt_sec=0.1 / 1000.0, eta=1 / 300.0, r0=20.0, tau_STDP_sec=0.12)[2][-1] - w_init for dt in delta_ts]
    y = np.array(dws) / w_init * 100.0
    x = delta_ts * 1000.0
    ymin = y.min()
    tol = 1e-12 * np.abs(ymin) + 1e-12
    min_idxs = np.where(np.isclose(y, ymin, atol=tol, rtol=0))[0]
    if min_idxs.size >= 2:
        i1, i2 = (min_idxs[0], min_idxs[-1])
        ax_plastic.plot(x[i1:i2 + 1], y[i1:i2 + 1], color=colors[i], lw=lt, label=f'{freq} Hz')
    else:
        ax_plastic.plot(x, y, color=colors[i], lw=lt, label=f'{freq} Hz')
ax_plastic.axhline(0, linestyle=':', color='k', lw=lt)
ax_plastic.set_xlabel('$\\Delta t$ (ms)', fontsize=ft)
ax_plastic.set_ylabel('$\\Delta w / w_0$ (%)', fontsize=ft)
ax_plastic.set_title('STDP Rule', fontsize=ft)
ax_plastic.legend(frameon=False, fontsize=ft - 3, loc=(0.1, 0.8))
for spine in ['top', 'right']:
    ax_plastic.tick_params(labelsize=ft - 1)
    ax_plastic.spines[spine].set_visible(False)




# Load the short, full-resolution intensity traces used in the paired-neuron panels.
N_E = 100
N_I = 100
T = 3.9
factor = np.sqrt(N_E)
wee = 20 / factor
wei = 40 / factor
wii = 50 / factor
wie = 50 / factor
c_x = 0.0
tau_r = 15.0 / 1000.0
tau_STDP = 100.0 / 1000.0
tau_ou = 10.0 / 1000.0
tau_wee = 300
tau_wei = 100.0
seed = 113
dt = 0.0002
num_points = int(T / dt)
time_full = np.linspace(0, T, num_points, endpoint=False)
loc = 'Fig1Rastordata/'
t_start1 = start1 + 0.255
t_end1 = end1 + 0.255
t_start2 = start2 + 0.2
t_end2 = end2 + 0.2
mask_left = (time_full >= t_start1) & (time_full <= t_end1)
mask_right = (time_full >= t_start2) & (time_full <= t_end2)
time_win_left = time_full[mask_left]
time_win_right = time_full[mask_right]
file_lamE = f'{loc}tracked_lamE_Ne_{N_E}_T{T}_wee{wee}_wei{wei}_wii{wii}_wie{wie}_sigma0.0_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
df_lamE = pd.read_csv(file_lamE)
neurons_left = (0, 5)
neurons_right = (0, 4)


# Extract a selected neuron and time window from the stored intensity matrix.
def slice_window(df, row, mask):
    arr = df.iloc[row, :].values
    return arr[:-1][mask]
lamE0_left = slice_window(df_lamE, neurons_left[0], mask_left)
lamE5_left = slice_window(df_lamE, neurons_left[1], mask_left)
axes_left[2].plot(time_win_left, lamE0_left, lw=lt, color='tab:green')
axes_left[3].plot(time_win_left, lamE5_left, lw=lt, color='tab:orange')
mask_right1 = (time_full >= t_start2 + 0.145) & (time_full <= t_end2 + 0.145)
lamE0_right = slice_window(df_lamE, neurons_right[0], mask_right1)
lamE5_right = slice_window(df_lamE, neurons_right[1], mask_right)
axes_right[2].plot(time_win_right, lamE0_right, lw=lt, color='tab:green')
axes_right[3].plot(time_win_right, lamE5_right, lw=lt, color='tab:orange')
for ax in (axes_left[4], axes_right[4]):
    ax.xaxis.labelpad = -6.0
for col_axes in [axes_left, axes_right]:
    pos_last = col_axes[4].get_position()
    shift = -0.05
    col_axes[4].set_position([pos_last.x0, pos_last.y0 - shift, pos_last.width, pos_last.height])


# Add the raster, mean-weight, and mean-rate panels in the rightmost column.
sub = gs[0:5, 4].subgridspec(nrows=3, ncols=1, height_ratios=[1.0, 1.0, 1.0], hspace=0.2)
ax_rastor = fig.add_subplot(sub[0, 0])
ax_weight = fig.add_subplot(sub[1, 0])
ax_rates = fig.add_subplot(sub[2, 0])

# Parameters that identify the seed-0 raster data files.
sigma = 0.0
ms = 1000.0
N_E = 500
N_I = 500
T_big = 10.0
loc = 'Fig1Rastordata/'
factor = np.sqrt(N_E)
wie = 0.02 / factor
wii = 0.02 / factor
wee0 = 0.036 / factor
wei0 = 0.036 / factor
seed = 0
c_x = 0.0
tau_STDP = 40.0 / ms
tau_ou = 8.0 / ms
tau_r = 8.0 / ms
tau_wee = 75000. 
tau_wei = 25000. 

# Load the external, excitatory, and inhibitory spike-time arrays.
fileX = f'{loc}spike_trains_X_Ne_{N_E}_T{T_big}_wee{wee0 * factor}_wei{wei0 * factor}_wii{wii * factor}_wie{wie * factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
fileE = f'{loc}spike_trains_E_Ne_{N_E}_T{T_big}_wee{wee0 * factor}_wei{wei0 * factor}_wii{wii * factor}_wie{wie * factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
fileI = f'{loc}spike_trains_I_Ne_{N_E}_T{T_big}_wee{wee0 * factor}_wei{wei0 * factor}_wii{wii * factor}_wie{wie * factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
dfX = pd.read_csv(fileX)
dfE = pd.read_csv(fileE)
dfI = pd.read_csv(fileI)

# Plot the selected asynchronous time window with population-specific offsets.
t0 = 9.825
t1 = 9.865
ax_rastor.set_xlim(t0, t1)
ax_rastor.set_title('Asynchronous Rastor', fontsize=ft)
N_X = dfX.shape[0]
Eoff = 0
Ioff = N_E
Xoff = N_E + N_I
half = 10.0
for i in range(N_E):
    st = dfE.iloc[i].values
    st = st[st > 0]
    y = Eoff + i
    ax_rastor.vlines(st, y - half, y + half, colors='tab:red', linewidth=0.3)
for i in range(N_I):
    st = dfI.iloc[i].values
    st = st[st > 0]
    y = Ioff + i
    ax_rastor.vlines(st, y - half, y + half, colors='tab:blue', linewidth=0.3)
for i in range(N_X):
    st = dfX.iloc[i].values
    st = st[st > 0]
    y = Xoff + i
    ax_rastor.vlines(st, y - half, y + half, colors='k', linewidth=0.3)
ax_rastor.set_xticks([t0, t1], [0, int((t1 - t0) * ms)], fontsize=ft)
ax_rastor.set_yticks([1, 2000], [1, 2000], fontsize=ft)
ax_rastor.set_xlabel('Time (ms)', fontsize=ft, labelpad=-10)
ax_rastor.set_ylabel('Neuron Idx', fontsize=ft, labelpad=-23)

# Parameters that identify the long plasticity simulations.
seeds = 15
skip = 1000
T = 100.0
N_E = 500
N_I = 500
ms = 1000.0
tau_STDP = 120.0 / ms
tau_ou = 8.0 / ms
tau_wei = 75000
tau_wee = 225000
dt = 0.1 / ms
factor = np.sqrt(N_E)
wie = 0.02 / factor
wii = 0.02 / factor
wee_init = 0.036 / factor
wei_init = 0.036 / factor
loc = 'Fig1data/'
time = np.linspace(0, T, int(T / dt))
time_reduced = time[::skip]
tau_r = 8.0 / ms
sigma = 0.0

# Allocate arrays for the seed-by-seed mean weights and population rates.
mean_wee_seeds = np.zeros((seeds - 1, len(time_reduced)))
mean_wei_seeds = np.zeros((seeds - 1, len(time_reduced)))
mean_re_seeds = np.zeros((seeds - 1, len(time_reduced)))
mean_ri_seeds = np.zeros((seeds - 1, len(time_reduced)))

# Load each realization into the corresponding row of the aggregate arrays.
for seed in range(1, seeds):
    df_wee_path = f'{loc}mean_ee_weights_Ne_{N_E}_T{T}_wee{wee_init * factor}_wei{wei_init * factor}_wii{wii * factor}_wie{wie * factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
    df_wei_path = f'{loc}mean_ei_weights_Ne_{N_E}_T{T}_wee{wee_init * factor}_wei{wei_init * factor}_wii{wii * factor}_wie{wie * factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
    df_rates_path = f'{loc}mean_rates_Ne_{N_E}_T{T}_wee{wee_init * factor}_wei{wei_init * factor}_wii{wii * factor}_wie{wie * factor}_sigma{sigma}_cx{c_x}_taur{tau_r}_tauSTDP{tau_STDP}_tauou{tau_ou}_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv'
    mean_wee_seeds[seed - 1, :] = pd.read_csv(df_wee_path).values[:-1, 0]
    mean_wei_seeds[seed - 1, :] = pd.read_csv(df_wei_path).values[:-1, 0]
    rates_array = pd.read_csv(df_rates_path).values
    mean_re_seeds[seed - 1, :] = rates_array[0, :-1]
    mean_ri_seeds[seed - 1, :] = rates_array[1, :-1]

# Compute across-seed means and standard deviations.
mean_wee = np.mean(mean_wee_seeds, axis=0)
std_wee = np.std(mean_wee_seeds, axis=0)
mean_wei = np.mean(np.abs(mean_wei_seeds), axis=0)
std_wei = np.std(np.abs(mean_wei_seeds), axis=0)
mean_re = np.mean(mean_re_seeds, axis=0)
std_re = np.std(mean_re_seeds, axis=0)
mean_ri = np.mean(mean_ri_seeds, axis=0)
std_ri = np.std(mean_ri_seeds, axis=0)

# Plot normalized mean synaptic weights with across-seed variability.
ax_weight.plot(time_reduced, mean_wee / mean_wee[0], c='tab:red', label='$\\bar{w}_{EE}$', lw=lt)
ax_weight.fill_between(time_reduced, mean_wee / mean_wee[0] - std_wee / mean_wee[0], mean_wee / mean_wee[0] + std_wee / mean_wee[0], color='tab:red', alpha=0.25)
ax_weight.plot(time_reduced, mean_wei / mean_wei[0], c='tab:blue', label='$\\bar{w}_{EI}$', lw=lt)
ax_weight.fill_between(time_reduced, mean_wei / mean_wei[0] - std_wei / mean_wei[0], mean_wei / mean_wei[0] + std_wei / mean_wei[0], color='tab:blue', alpha=0.25)
ax_weight.set_xticks([], [])
ax_weight.set_ylabel('$\\bar{w}/\\bar{w}_0$', fontsize=ft, labelpad=-5)
ax_weight.legend(fontsize=ft - 2, loc='upper right', frameon=False)
ax_weight.set_ylim(0.0, 1.2)
ax_weight.set_yticks([0, 1], [0, 1], fontsize=ft)
ax_weight.set_xlim(0, 70)

# Plot excitatory and inhibitory population rates with across-seed variability.
ax_rates.set_yticks([20], [20], fontsize=ft)
ax_rates.set_xticks([0, 70], [0, 70], fontsize=ft)
ax_rates.set_ylim(0, 25)
ax_rates.plot(time_reduced, mean_re, c='tab:red', label='$\\bar{r}_E$', lw=lt)
ax_rates.fill_between(time_reduced, mean_re - std_re, mean_re + std_re, color='tab:red', alpha=0.25)
ax_rates.plot(time_reduced, mean_ri, c='tab:blue', label='$\\bar{r}_I$', lw=lt)
ax_rates.fill_between(time_reduced, mean_ri - std_ri, mean_ri + std_ri, color='tab:blue', alpha=0.25)
ax_rates.set_xlim(0, 70)
ax_rates.set_xlabel('Time (sec)', fontsize=ft, labelpad=-5)
ax_rates.set_ylabel('$\\bar{r}$ (Hz)', fontsize=ft, labelpad=-5)
ax_rates.yaxis.set_label_coords(-0.05, 0.3)
ax_rates.legend(fontsize=ft, loc='lower right', frameon=False)

# Standardize line widths, spines, and tick labels across all panels.
for ax in fig.axes:
    for line in ax.get_lines():
        line.set_linewidth(lt)
    for side in ['left', 'bottom']:
        if side in ax.spines:
            ax.spines[side].set_linewidth(SPINE_W)
    ax.tick_params(labelsize=ft - 1)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

# Apply the final panel positions and save the assembled figure.
plt.subplots_adjust(hspace=0.1, wspace=0.35, left=0.08, right=0.97, top=0.75, bottom=0.09)
for ax, dist in zip([ax_rastor, ax_weight, ax_rates], [0.14, 0.07, 0.0]):
    axx = ax.get_position()
    shift = dist
    ax.set_position([axx.x0 + 0.01, axx.y0 + shift, axx.width, axx.height * 1.2])
pos_plastic = ax_plastic.get_position()
shift = -0.3
ax_plastic.set_position([pos_plastic.x0, pos_plastic.y0 + shift, pos_plastic.width, pos_plastic.height * 0.8])
TITLE_FS = 12
TICK_FS = 8
mpl.rcParams.update({'xtick.labelsize': TICK_FS, 'ytick.labelsize': TICK_FS, 'axes.labelsize': TITLE_FS})
#plt.savefig('Fig1.pdf', dpi=150)

# Uncomment for interactive inspection.
plt.show()
