#!/usr/bin/env python3
"""Plot the three panels in the third row of Figure 2."""

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re as _re

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / 'Figure2Row3Data'
OUTPUT_FILE = BASE_DIR / 'Figure2Row3.pdf'

# Font settings retained from the original figure script.
TITLE_FS = 11
LABEL_FS = 11
TICK_FS = 8
LEGEND_FS = 9
lt, ft = 1., 8

mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'text.latex.preamble': r'\usepackage{amsmath}'})

# Parameters encoded in the simulation filenames.
N = 500
ms_per_sec = 1000
tau_rprim = 2.0 / ms_per_sec
scale_factor = np.sqrt(N)
factor = np.sqrt(N)
SIMLEN_PULSES = 210.


def sc(raw):
    """Convert the raw weight notation to the simulation units."""
    return raw / scale_factor * tau_rprim


w_IE = sc(10.) * factor
w_II = sc(10.) * factor
w_IX = sc(15.)
sigma = 0
c_x = 0
tau_r = 8. / ms_per_sec
tau_ou = 8. / ms_per_sec
tau_STDP = 120. / ms_per_sec
tau_wee = 225000.
tau_wei = 75000.
WEIGHTS_RAW = dict(PULSE=dict(w_EE=15., w_EI=15.))

_RX_WEE_WEI = _re.compile(r'_wee([-+0-9.eE]+)_wei([-+0-9.eE]+)_')


def _tok(x, places=19):
    """Format floating-point filename tokens consistently."""
    d = Decimal(str(float(x))).quantize(Decimal(f'1e-{places}'), rounding=ROUND_HALF_UP)
    s = format(d, 'f').rstrip('0').rstrip('.')
    return s or '0'


def build_path(kind, wee_sc, wei_sc, seed, Tlen, tol=1e-12):
    """Build the original CSV filename, with the original numeric fallback."""
    fname = (f'{kind}Ne_{N}_T{float(Tlen)}_wee{_tok(wee_sc)}_wei{_tok(wei_sc)}'
             f'_wii{_tok(w_II)}_wie{_tok(w_IE)}_sigma{_tok(sigma)}_cx{c_x}'
             f'_taur{_tok(tau_r)}_tauSTDP{_tok(tau_STDP)}_tauou{tau_ou}'
             f'_tauwee{tau_wee}_tauwei{tau_wei}_seed{seed}.csv')
    path = data_dir / fname
    if path.exists():
        return path

    if 'rates' in kind:
        pattern = data_dir / f'{kind}Ne_{N}_T{Tlen}_wee*_wei*_*_seed{seed}.csv'
    else:
        pattern = data_dir / f'{kind}{N}_T{Tlen}_wee*_wei*_*_seed{seed}.csv'

    candidates = [candidate for candidate in pattern.parent.glob(pattern.name) if _RX_WEE_WEI.search(candidate.name)]
    if not candidates:
        return path

    target_wee = float(wee_sc)
    target_wei = float(wei_sc)

    def dist(candidate):
        a, b = map(float, _RX_WEE_WEI.search(candidate.name).groups())
        return abs(a - target_wee) + abs(b - target_wei)

    best = min(candidates, key=dist)
    return best if dist(best) <= tol else path


def load_stack(kind_prefix, wee_raw, wei_raw, Tlen, seeds, mode='rates', weighted=False):
    """Load and stack the original rate or weight CSVs across seeds."""
    if weighted:
        wee_sc, wei_sc = wee_raw, wei_raw
    else:
        wee_sc, wei_sc = sc(float(wee_raw)), sc(wei_raw)

    stacks = []
    for seed in seeds:
        arr = pd.read_csv(build_path(kind_prefix, wee_sc, wei_sc, seed, Tlen)).values
        if mode == 'weights':
            stacks.append(arr[:, 0])
        elif mode == 'rates':
            stacks.append(arr[:, :-1])
    return np.stack(stacks, axis=0)


def reference_path(filename):
    """Locate a line-attractor reference file without changing its contents."""
    for path in (BASE_DIR / filename, data_dir / filename):
        if path.exists():
            return path
    raise FileNotFoundError(f'Missing {filename}. Place the original file beside this script or in {data_dir.name}/.')


def main():
    """Load the multi-pulse outputs and reproduce the three selected panels."""
    # Keep the original grid positions so the panel geometry is unchanged.
    fig = plt.figure(figsize=(7.5, 4.5), dpi=200)
    grid = fig.add_gridspec(2, 5, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1])
    ax4 = fig.add_subplot(grid[1, 3])
    ax5 = fig.add_subplot(grid[1, 1])
    sub = grid[1, 2].subgridspec(2, 1, hspace=0.3)
    ax6 = fig.add_subplot(sub[0, 0])
    ax7 = fig.add_subplot(sub[1, 0])

    # Dashed line-attractor reference shown in the original weight-space panel.
    final_wEE_list = np.loadtxt(reference_path('final_wEE_list_simulation.txt'))
    final_wEI_list = np.loadtxt(reference_path('final_wEI_list_simulation.txt'))
    ax4.plot(final_wEE_list / scale_factor * tau_rprim, final_wEI_list / scale_factor * tau_rprim, color='r', linestyle='--')

    # Load the multi-pulse simulation output used by all three panels.
    wraw = WEIGHTS_RAW['PULSE']
    seeds_p = [0]
    re_p = load_stack('multi_pulse_mean_rates_', np.around(sc(wraw['w_EE']) * factor, 3), np.around(sc(wraw['w_EI']) * factor, 3), SIMLEN_PULSES, seeds_p, weighted=True)[:, 0]
    ri_p = load_stack('multi_pulse_mean_rates_', np.around(sc(wraw['w_EE']) * factor, 3), np.around(sc(wraw['w_EI']) * factor, 3), SIMLEN_PULSES, seeds_p, weighted=True)[:, 1]
    w_ee_p = load_stack('multi_pulse_mean_ee_weights_', np.around(sc(wraw['w_EE']) * factor, 3), np.around(sc(wraw['w_EI']) * factor, 3), SIMLEN_PULSES, seeds_p, weighted=True, mode='weights').squeeze()
    w_ei_p = load_stack('multi_pulse_mean_ei_weights_', np.around(sc(wraw['w_EE']) * factor, 3), np.around(sc(wraw['w_EI']) * factor, 3), SIMLEN_PULSES, seeds_p, weighted=True, mode='weights').squeeze()

    t_p_csv = np.linspace(0, SIMLEN_PULSES, re_p.shape[1])
    w_ee_mean = w_ee_p
    w_ei_mean = w_ei_p

    # Right panel: trajectory of the mean synaptic weights during learning.
    points_ws = np.array([w_ee_mean[:-2], np.abs(w_ei_mean)[:-2]]).T.reshape(-1, 1, 2)
    seg_ws = np.concatenate([points_ws[:-1], points_ws[1:]], axis=1)
    lc_ws = LineCollection(seg_ws, cmap='plasma', norm=plt.Normalize(t_p_csv.min(), t_p_csv.max()))
    lc_ws.set_array(t_p_csv[:len(seg_ws)])
    lc_ws.set_linewidth(lt)
    ax4.add_collection(lc_ws)
    ax4.autoscale()
    ax4.set_ylim(0.0009, .003)
    ax4.set_xlim(0.0008, .0025)
    ax4.set_yticks([0.001, .003], ['.001', '.003'])
    ax4.set_xticks([0.001, .0025], ['.001', '.0025'])
    ax4.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3), useMathText=True)
    ax4.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax4.ticklabel_format(axis='x', style='sci', scilimits=(-3, -3), useMathText=True)
    ax4.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)
    ax4.xaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)

    # Left panel: mean excitatory population activity during learning.
    points_re_sim = np.array([t_p_csv, re_p.mean(0)]).T.reshape(-1, 1, 2)
    seg_re_sim = np.concatenate([points_re_sim[:-1], points_re_sim[1:]], axis=1)
    lc_re_sim = LineCollection(seg_re_sim, cmap='plasma', norm=plt.Normalize(t_p_csv.min(), t_p_csv.max()))
    lc_re_sim.set_array(t_p_csv[:-1])
    lc_re_sim.set_linewidth(lt)
    ax5.add_collection(lc_re_sim)
    ax5.set_ylim(0, 100)
    ax5.set_yticks([0, 100], [0, 100])
    ax5.set_xticks([], [])
    ax5.set_xlim(0, 200)

    # Middle panel, top: normalized mean E-to-E weight.
    points_t = np.array([t_p_csv, w_ee_mean[:-1] / w_ee_mean[0]]).T.reshape(-1, 1, 2)
    segments_t = np.concatenate([points_t[:-1], points_t[1:]], axis=1)
    lc_t = LineCollection(segments_t, cmap='plasma', norm=plt.Normalize(t_p_csv.min(), t_p_csv.max()))
    lc_t.set_array(t_p_csv[:-1])
    lc_t.set_linewidth(lt)
    ax6.add_collection(lc_t)
    ax6.autoscale()
    ax6.set_ylabel(r'$\bar{w}_{EE}/\bar{w}_{EE0}$', labelpad=-5, fontsize=LEGEND_FS)
    ax6.set_title(r'$\bar{w}$ vs Time ', fontsize=ft)
    ax6.set_ylim(.95, 2.01)
    ax6.set_yticks([1, 2], ['1', '2'])
    ax6.set_xticks([0, 200], ['0', '200'])
    ax6.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax6.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)

    # Middle panel, bottom: normalized mean I-to-E weight magnitude.
    points_t = np.array([t_p_csv, np.abs(w_ei_mean[:-1] / w_ei_mean[0])]).T.reshape(-1, 1, 2)
    segments_t = np.concatenate([points_t[:-1], points_t[1:]], axis=1)
    lc_t = LineCollection(segments_t, cmap='plasma', norm=plt.Normalize(t_p_csv.min(), t_p_csv.max()))
    lc_t.set_array(t_p_csv[:-1])
    lc_t.set_linewidth(lt)
    ax7.add_collection(lc_t)
    ax7.autoscale()
    ax7.set_xlabel('Time (s)', labelpad=-10, fontsize=LEGEND_FS)
    ax7.set_ylabel(r'$\bar{w}_{EI}/\bar{w}_{EI0}$', labelpad=-5, fontsize=LEGEND_FS)
    ax7.set_ylim(.95, 2.01)
    ax7.set_yticks([1., 2.], ['1', '2'])
    ax7.set_xticks([0, 200], ['0', '200'])
    ax7.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax7.yaxis.get_offset_text().set_fontsize(TICK_FS * 0.8)

    # Titles and labels retained from the selected axes in Fig3.py.
    ax5.set_title(r'$\bar{r}_E$ during Learning', fontsize=ft)
    ax5.set_xlabel('Time (s)', fontsize=LEGEND_FS)
    ax5.set_ylabel(r'$\bar{r}_E(t)$  (Hz)', fontsize=ft, labelpad=-10)
    ax4.set_title(r'$\bar{w}$ during Learning', fontsize=ft)
    ax4.set_xlabel(r'$w_{EE}$', fontsize=ft, labelpad=-5)
    ax4.set_ylabel(r'$w_{EI}$', fontsize=ft, labelpad=0)

    # Apply the original axis styling only to the selected panels.
    for ax in (ax4, ax5, ax6, ax7):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(lt)
        ax.spines['bottom'].set_linewidth(lt)
        ax.tick_params(labelsize=ft - 1)

    plt.rcParams.update({'axes.titlesize': TITLE_FS, 'axes.labelsize': LABEL_FS, 'xtick.labelsize': TICK_FS, 'ytick.labelsize': TICK_FS, 'legend.fontsize': LEGEND_FS})

    plt.subplots_adjust(hspace=.35, bottom=.08, top=.92, left=.05, wspace=.42, right=.98)
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
        ax.tick_params(axis='both', which='minor', labelsize=TICK_FS)

    # Crop away the unused grid cells while keeping the original panel geometry.
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)
    print(f'Saved {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
