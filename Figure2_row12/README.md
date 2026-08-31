
# Figure 2: spiking-network trajectories and line-attractor analysis

## Preservation statement

**This cleanup does not change the simulation, the analysis, or what gets plotted.** The active parameter values, random-number calls and order, plasticity equations, recorded arrays, CSV filename conventions, data loading, averaging, regression, error calculations, plotting calls, plotting order, colors, line styles, labels, limits, ticks, and layout are retained from the supplied files.

The changes are limited to:

- organizing each script into clearly commented sections;
- consolidating duplicate imports;
- removing overwritten assignments, inactive commented-out alternatives, and calculations that had no downstream use;
- formatting individual function calls on one physical line where practical;
- preserving the original variable names and notation rather than introducing renamed constants or a new program structure.

## Scientific role in Figure 2

These scripts support the Figure 2 analysis of plastic E/I network trajectories initiated from different excitatory-weight conditions. The plotting script shows:

1. mean E-to-E weight, $\bar{w}_{EE}$, over time;
2. mean I-to-E weight, $\bar{w}_{EI}$, over time;
3. trajectories in $(\bar{w}_{EE},\bar{w}_{EI})$ weight space and a linear fit to their final coordinates;
4. mean excitatory firing-rate trajectories for the highest and lowest plotted initial conditions;
5. late-time mean inhibitory and excitatory firing rates as a function of normalized position along the fitted line;
6. late-time excitatory-rate variance as a function of normalized position along the fitted line.


## Repository contents

| File | Role |
|---|---|
| `RunFig2Spiking.py` | Runs the supplied parameter sweep and writes simulation summaries to CSV files. |
| `Fig2SpikingSim.py` | Implements the spiking E/I simulation, plasticity updates, and reduced-time recording. |
| `helper_functions.py` | Provides the recurrent/external weight constructors used by the simulation and retains the supplied auxiliary covariance utilities. |
| `Fig2.py` | Loads the required CSV files, performs the retained line-attractor and rate analyses, and constructs the Figure 2 panels. |

## Workflow

### 1. Generate spiking simulation data

The supplied simulation entry point is:

```bash
mkdir -p NewFig2data
python RunFig2Spiking.py
```

### 2. Construct Figure 2

Place the CSV files expected by the plotting script in:

```text
Fig2data/
```

Then run:

```bash
python Fig2.py
```

