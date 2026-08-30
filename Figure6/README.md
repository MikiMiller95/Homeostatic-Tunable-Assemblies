[README(6).md](https://github.com/user-attachments/files/31622517/README.6.md)
# Figure 6: correlated-input drift in a plastic E–I network

This directory contains the simulation, theory, and plotting code associated with **Figure 6**. The files were cleaned and relabeled from the supplied Figure 7 versions without changing the active simulation equations, analyses, parameter values, random-seed mappings, plotting calls, panel assignments, axis settings, or plotted quantities.

The cleanup was deliberately conservative:

- Figure 7 filenames, module imports, data prefixes, and the final PDF name were relabeled for Figure 6.
- Duplicate and unused imports were removed.
- Stale commented-out code and inactive triple-quoted plotting blocks were removed.
- Long function calls were placed on one line where possible.
- Comments and section headings were added around the existing code.
- Active but apparently unused variables and outputs were retained rather than removed.
- Pre-existing inconsistencies were documented below rather than corrected.

## Directory contents

| File | Role | Run directly? | Required for `Fig6.pdf`? |
|---|---|---:|---:|
| `Fig6.py` | Loads all precomputed CSV files, assembles the six plotted panels, and saves `Fig6.pdf`. | Yes, last | Yes |
| `RunFig6Spiking.py` | Intended driver for the main correlation-strength sweep saved under `Fig6data/`. | Yes | Yes, but see blockers below |
| `Fig6SpikingSim.py` | Spiking simulation and reciprocity calculation used by `RunFig6Spiking.py`. | No | Yes |
| `RunTheoryAgain.py` | Intended driver for the mean-field theory trajectories saved under `Fig6data/`. | Yes | Yes, but `Fig6D.py` is missing |
| `RunDrift6D.py` | Generates the long-timescale weight and rate traces saved under `Driftdata/`. | Yes | Yes |
| `Fig6DriftSim.py` | Spiking simulation used by `RunDrift6D.py`. | No | Yes |
| `CorrelatedRastor.py` | Generates the external, excitatory, and inhibitory spike-train CSVs for the correlated-input raster panel. The original `Rastor` spelling was retained because the plotted title was not changed. | Yes | Yes |
| `helper_functions.py` | Shared covariance and connectivity functions imported by the simulation modules. | No | Yes |
| `Fig6data/` | Expected location of the main sigma-sweep simulation and theory CSVs. | — | Yes |
| `Driftdata/` | Expected location of the long-trace and raster CSVs. | — | Yes |

## File excluded from this repository

`BalanceIndex.py` was not included. It is a standalone two-panel balance-index visualization. None of the retained scripts imports it, it does not generate a CSV loaded by `Fig6.py`, and it is not part of the simulation-to-figure dependency chain.

## Python dependencies

Install the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

A separate local module named `Fig6D.py` is also required by `RunTheoryAgain.py`. The supplied theory driver originally imported `Fig7D.py`, but that file was not among the supplied files.

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
