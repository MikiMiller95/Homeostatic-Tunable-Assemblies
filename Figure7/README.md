[README.md](https://github.com/user-attachments/files/31622561/README.md)
# Figure 7: correlated E/I input and synaptic drift

This directory contains the code associated with **Figure 7**. The figure compares spiking-network simulations with population theory to show how correlations shared across excitatory and inhibitory external inputs shape synaptic drift.

The active plotting script fills four panels of a 2 × 3 canvas:

- the upper-right panel shows excitatory, inhibitory, and external-input spike rasters around the switch from independent to correlated input;
- the lower-left panel places simulated weight trajectories in the recurrent-weight plane together with the line attractor and stability boundaries;
- the lower-middle panel compares simulated and theoretical mean E-to-E weight trajectories for selected input-correlation values;
- the lower-right panel summarizes weight change as a function of E/I input correlation.

The upper-left and upper-middle axes are intentionally left blank for manuscript schematics that are not drawn by this Python script.

## Repository contents

| File | Role | Run directly? |
|---|---|---:|
| `Fig7.py` | Loads the precomputed CSV files, performs the existing Figure 7 aggregation, and saves `Fig7.pdf`. | Yes, last |
| `CorrelatedRaster.py` | Runs the short correlation-switch spiking simulation and writes the three spike-train CSV files used by the raster panel. | Yes |
| `RunFig7Theory.py` | Runs the deterministic two-population theory across the input-correlation sweep and writes theory weight trajectories. | Yes |
| `Fig7D.py` | Defines the rate-and-weight theory integrator called by `RunFig7Theory.py`. | No |
| `helper_functions.py` | Defines the covariance-spectrum and connectivity helpers used by the simulation and theory scripts. | No |
| `Fig7data/` | Expected location of the long simulation and theory CSV files. | — |
| `Driftdata/` | Expected location of the raster spike-train CSV files. | — |

`requirements.txt` lists the Python packages imported by these files. The empty data directories are retained with `.gitkeep` files so that a fresh clone has the expected paths.

```text
Fig7data/mean_ee_weights_..._seed<S>.csv
Fig7data/mean_ei_weights_..._seed<S>.csv
```

## Usage

### 1. Install dependencies

From this directory:

```bash
python -m pip install -r requirements.txt
```

### 2. Generate the raster data

```bash
python CorrelatedRaster.py
```

This runs the 7.5 s, 1,000-E/1,000-I correlation-switch simulation and writes:

```text
Driftdata/spike_trains_E_..._seed530.csv
Driftdata/spike_trains_I_..._seed530.csv
Driftdata/spike_trains_X_..._seed530.csv
```

The plotted raster marks the input-correlation switch at 6.8 s. The script deliberately preserves the original update order, in which the newly assigned correlation value affects the next integration step.

### 3. Generate the theory trajectories

```bash
python RunFig7Theory.py
```

This calls `Fig7D.run_sim()` for all six correlation values and writes seed-tagged theory trajectories to `Fig7data/`.

The script writes both:

```text
Fig7data/W_theory_..._seed<S>.csv
Fig7data/Wei_theory_..._seed<S>.csv
```

### 4. Create the figure

```bash
python Fig7.py
```

The script reads the long-simulation data from `Fig7data/`, the raster data from `Driftdata/`, and writes:

```text
Fig7.pdf
```
