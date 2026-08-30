[README.md](https://github.com/user-attachments/files/31622658/README.md)
# Figure 6: correlated-input drift in a plastic E–I network

This directory contains the simulation, theory, and plotting code associated with **Figure 6**. The files were cleaned and relabeled from their supplied Figure 7 versions without changing the active simulation equations, analyses, random-seed mappings, plotting calls, panel assignments, axis settings, or plotted quantities.

The following user-specified regeneration corrections are included:

- `RunFig6Spiking.py` now defines `tau_rprim = 2.0 / ms_per_sec` before constructing the recurrent weights.
- `RunFig6Spiking.py` now runs `sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`.
- The supplied `Fig7D(2).py` theory implementation was cleaned, relabeled as `Fig6D.py`, and connected to `RunTheoryAgain.py`.

The remaining cleanup was deliberately conservative:

- Figure 7 filenames, module imports, data prefixes, and the final PDF name were relabeled for Figure 6.
- Duplicate and unused imports were removed.
- Stale commented-out code and inactive triple-quoted plotting blocks were removed.
- Long function calls were placed on one line where practical.
- Comments and section headings were added around the existing code.
- Active but potentially consequential calculations were retained rather than silently removed.
- Remaining pre-existing inconsistencies are documented below rather than corrected.

## Directory contents

| File | Role | Run directly? | Required for `Fig6.pdf`? |
|---|---|---:|---:|
| `Fig6.py` | Loads all precomputed CSV files, assembles the plotted panels, and saves `Fig6.pdf`. | Yes, last | Yes |
| `RunFig6Spiking.py` | Runs the main correlation-strength spiking sweep and saves outputs under `Fig6data/`. | Yes | Yes, but see the remaining seed and filename issues below |
| `Fig6SpikingSim.py` | Spiking simulation and reciprocity calculation used by `RunFig6Spiking.py`. | No | Yes |
| `RunTheoryAgain.py` | Runs the mean-field theory sweep and saves theory trajectories under `Fig6data/`. | Yes | Yes |
| `Fig6D.py` | Mean-field rate and plasticity integration used by `RunTheoryAgain.py`. | No | Yes |
| `RunDrift6D.py` | Generates the long-timescale weight and rate traces saved under `Driftdata/`. | Yes | Yes |
| `Fig6DriftSim.py` | Spiking simulation used by `RunDrift6D.py`. | No | Yes |
| `CorrelatedRastor.py` | Generates the external, excitatory, and inhibitory spike-train CSVs for the correlated-input raster panel. The original `Rastor` spelling was retained because changing it is unnecessary for regeneration. | Yes | Yes |
| `helper_functions.py` | Shared covariance and connectivity functions imported by the simulation and theory modules. | No | Yes |
| `Fig6data/` | Expected location of the main sigma-sweep simulation and theory CSVs. | — | Yes |
| `Driftdata/` | Expected location of the long-trace and raster CSVs. | — | Yes |

## File excluded from this repository

`BalanceIndex.py` was not included. It is a standalone two-panel balance-index visualization. None of the retained scripts imports it, it does not generate a CSV loaded by `Fig6.py`, and it is not part of the simulation-to-figure dependency chain.

## Python dependencies

Install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Data used by `Fig6.py`

`Fig6.py` combines three independent data products.

### 1. Main sigma sweep: `Fig6data/`

The main weight-space, weight-trajectory, and integrated-drift panels expect:

- correlation strengths `sigma = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`;
- seeds `0` through `19`;
- `mean_ee_weights_...csv` and `mean_ei_weights_...csv` for every sigma and seed;
- `W_theory_...csv` for every sigma and seed;
- `balance_idx_...csv` and `q_val_...csv` for `sigma = 0.0` and `0.5`, even though the plotting block that used these quantities is inactive in the supplied figure code.

The intended generators are:

```bash
python RunFig6Spiking.py
python RunTheoryAgain.py
```

`RunFig6Spiking.py` now covers all six sigma values but still runs only five seeds. `RunTheoryAgain.py` runs 23 seeds; `Fig6.py` uses the first 20 theory files.

### 2. Long weight and rate traces: `Driftdata/`

The long-timescale panels expect simulation data for:

- `sigma = 0.0` and `0.5`;
- seeds `0` through `9`;
- `T = 80.0` seconds;
- mean E-to-E weights, mean I-to-E weights, and mean E/I rates.

Generate these files with:

```bash
python RunDrift6D.py
```

`RunDrift6D.py` runs 20 seeds, while `Fig6.py` reads only the first 10. It also writes balance and theory files that are not read by the final figure script.

### 3. Correlated-input raster: `Driftdata/`

The raster panel expects external, excitatory, and inhibitory spike trains for:

- `sigma = 4.0`;
- simulation seed `54000`;
- `T = 7.0` seconds;
- the displayed interval surrounding the correlation onset at `6.8` seconds.

Generate these files with:

```bash
python CorrelatedRastor.py
```

## Intended run order

The first four commands generate independent data products and may be run in any order. Run the plotting script only after all expected CSV files are present.

```bash
python RunFig6Spiking.py
python RunTheoryAgain.py
python RunDrift6D.py
python CorrelatedRastor.py
python Fig6.py
```

The final command saves:

```text
Fig6.pdf
```

## Remaining pre-existing regeneration blockers not fixed

The supplied files still do **not** form a complete end-to-end regeneration pipeline without additional corrections. These issues were left unchanged so that the simulations, analyses, and plotted content were not silently altered.

1. **The main simulation still generates too few seeds.** `RunFig6Spiking.py` uses `seeds = 5`, while `Fig6.py` loads seeds `0` through `19` for each of the six sigma values.

2. **The main weight filenames use inconsistent plasticity timescales.** `RunFig6Spiking.py` saves its mean-weight files with `tauwee=225000` and `tauwei=75000`, whereas `Fig6.py` loads those files using `tauwee=450` and `tauwei=150`. The balance and reciprocity filenames in `Fig6.py` use the larger values, so this is not a single global rename.

3. **`balance_index` is returned as `None` by `Fig6SpikingSim.py`.** `RunFig6Spiking.py` writes that value to CSV, and `Fig6.py` later attempts to read a numeric first column from the file.

Because of these remaining blockers, the files were syntax-checked but the full simulations and final PDF were not regenerated as part of this cleanup.

## Additional review notes not changed

These are possible inconsistencies or cleanup candidates, not corrections applied to the code:

- `Fig6.py` first assigns the integrated-drift title to `ax[1, 3]` and later overwrites that same title with `Mean wEE/w0 over Time`. Verify which title belongs on `ax[1, 2]` and which belongs on `ax[1, 3]`.
- Text is added to `ax[1, 3]` using `ax[1, 2].transAxes` as its transform. Verify that the annotations are intended to use the neighboring axis coordinate system.
- `Fig6.py` still loads `balance_idx` and `q_val` files even though the supplied plotting block that used them was inactive. Removing those reads would change the current file requirements, so they were retained.
- `CorrelatedRastor.py` uses `tau_rprim` as a module-level global inside `spiking_sim()` rather than passing it as an argument. It works only when the file is run through its current top-level setup.
- `generate_bernouilli_weight_array()` writes the nominal I-to-I block into `weights[:N_I, :N_I]`, which overlaps the upper-left block when `N_E == N_I`. The current simulations set `bernouilli = False`, so this branch is inactive for Figure 6.
- `calculate_spike_count_covariance()` calls `bin_spike_trains()`, but that function is not defined in `helper_functions.py`. This helper is not called by the retained Figure 6 workflow.
- `generate_weight_array()` uses `num_E` for both row and column block boundaries. That is consistent with the current `N_E = N_I = 500` simulations but should be reviewed before using unequal population sizes.
- Several legacy counters, event-tracking arrays, plotting handles, and returned arrays appear unused. They were left active because removing executable statements could obscure whether they were intentionally retained for diagnostics.

## Validation performed

Every retained Python file passes `python -m py_compile`. The supplied `Fig7D(2).py` theory module was also compared with the cleaned `Fig6D.py` at the statement level after ignoring imports, comments/docstrings, and formatting-only changes.
