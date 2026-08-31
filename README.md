[Repository_README.md](https://github.com/user-attachments/files/31619768/Repository_README.md)


# Homeostatic-Tunable-Assemblies
This is the primary repository for the paper **Homeostatic Plasticity Enables Stable, Flexible, and Tunable Assemblies** by Michelle C Miller, Christoph Miehl, and Brent Doiron. Code and analysis for studying how excitatory Hebbian and inhibitory homeostatic plasticity generate tunable neural assemblies, synaptic continua, and stable firing rates in recurrent neural networks.
# Homeostatic Plasticity and Tunable Neural Assemblies

This repository contains the simulation, analysis, and figure-generation code for a study of how excitatory Hebbian plasticity and inhibitory homeostatic plasticity produce tunable neural assemblies. The models examine how coupled plasticity can maintain a stable excitatory firing-rate target while allowing synaptic strength, network gain, timescales, and variability to change.

## Repository structure

The repository is organized by manuscript figure. Each figure folder contains its own README, the scripts used to generate the underlying data, and the script used to assemble the final figure.

```
homeostatic-tunable-assemblies/
├── README.md
├── Figure1/
│   ├── README.md
│   ├── Fig1.py
│   ├── Fig1B.py
│   ├── Fig1SpikingSim.py
│   ├── RunFig1Spiking.py
│   ├── Rastor.py
│   └── helper_functions.py
├── Figure2/
├── Figure3/
└── ...
```

Generated data may be excluded from Git when the files are large, provided that the corresponding simulation scripts and reproduction instructions are included.

## Requirements

The code uses Python 3 with NumPy, pandas, and Matplotlib.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy pandas matplotlib
```

## Reproducing a figure

Enter the relevant figure folder, follow its README to generate the required data, and then run the figure-assembly script. Because the scripts use relative paths, run them from inside their figure folder.

For Figure 1, see [`Figure1/README.md`](Figure1/README.md).

## Citation

Citation information for the associated manuscript will be added here.
