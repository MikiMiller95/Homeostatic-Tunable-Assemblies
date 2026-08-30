[Figure1_README.md](https://github.com/user-attachments/files/31619792/Figure1_README.md)
# Figure 1

This folder contains the simulation and plotting code for Figure 1. Run all commands from inside this folder.

## Files

- `Rastor.py` generates the excitatory, inhibitory, and external spike trains used in the asynchronous raster.
- `Fig1SpikingSim.py` defines the long spiking-network simulation.
- `RunFig1Spiking.py` runs the long simulations used for the mean weights and firing rates.
- `Fig1.py` loads the generated data, creates the STDP and network panels, and saves `Fig1.pdf`.
- `Fig1B.py` contains a separate rate-based plasticity-rule plotting helper; the current `Fig1.py` does not call it.
- `helper_functions.py` contains shared connectivity and analysis utilities.

## Procedure

Create the data folders:

```bash
mkdir -p Fig1Rastordata Fig1data_TauSTDP120ms
```

Generate the raster data:

```bash
python3 Rastor.py
```

Generate the long plasticity simulations:

```bash
python3 RunFig1Spiking.py
```

Before running this step, make the output path in `RunFig1Spiking.py` match the path read by `Fig1.py`. The plotting script expects the files in `Fig1data_TauSTDP120ms/`, while the current runner writes them under `NewNewFig1data/`.

Finally, assemble the figure:

```bash
python3 Fig1.py
```

The output is `Fig1.pdf`.

## Required input not generated here

`Fig1.py` also loads a full-resolution `tracked_lamE_*.csv` file from `Fig1data_TauSTDP150ms/` for the paired-neuron intensity traces. The uploaded Figure 1 scripts do not generate this file, so it must be included with the repository or produced by an additional short-simulation script. Despite the legacy folder name, the filename constructed by `Fig1.py` uses `tau_STDP = 0.1` seconds, or 100 ms.
