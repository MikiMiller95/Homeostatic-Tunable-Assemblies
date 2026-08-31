# Figure 5

This directory contains the cleaned and relabeled code for the figure that was previously stored as Figure 6. The final plotting script now saves `Fig5.pdf`.


The cleanup changes filenames, imports, formatting, comments, and code organization. It does **not** intentionally change the simulation equations, numerical parameters, active analyses, plotted quantities, panel layout, axis settings, or plotting calls.

## Directory contents

```text
Figure5_cleaned/
├── Fig5.py
├── Fig5Sim.py
├── Fig5SpikingSim.py
├── helper_functions.py
├── RunFig5SinglePulse.py
├── NewDatadata/
│   └── .gitkeep
└── README.md
```

### Files run directly

- `RunFig5SinglePulse.py` runs the single-pulse spiking simulations and writes the CSV files loaded by `Fig5.py`.
- `Fig5.py` loads the spiking CSVs, evaluates the deterministic theory and covariance calculations, assembles every panel, and saves `Fig5.pdf`.

### Imported support modules

- `Fig5SpikingSim.py` contains the spiking-network simulation called by `RunFig5SinglePulse.py`.
- `Fig5Sim.py` contains the deterministic two-population pulse-response simulation called by `Fig5.py`.
- `helper_functions.py` contains the connectivity and covariance utilities used by the simulation and plotting code.

The support modules are not normally run directly.

## Python dependencies

```text
numpy
pandas
matplotlib
scikit-learn
```

## Run order

Run the commands from this directory so the relative imports and data paths resolve correctly.

### 1. Generate the spiking simulation data

```bash
python RunFig5SinglePulse.py
```

The runner writes five CSV types to `data/` for each seed and initial `w_EE` value:

```text
one_pulse_mean_rates_...csv
one_pulse_track_weights_...csv
one_pulse_mean_ee_weights_...csv
one_pulse_mean_ei_weights_...csv
one_pulse_balance_idx_...csv
```


### 2. Create the final figure

```bash
python Fig5.py
```

This script:

1. Calls `run_sim()` from `Fig5Sim.py` for the deterministic pulse-response traces.
2. Loads the spiking mean-rate and balance-index CSV files from `NewDatadata/`.
3. Calls `calculate_CSD()` from `helper_functions.py` for the theoretical variance and autocovariance calculations.
4. Saves the completed figure as:

```text
Fig5.pdf
```
