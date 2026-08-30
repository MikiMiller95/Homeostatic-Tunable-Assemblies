# Figure 4

This directory contains the code used to generate `Fig5.pdf`. The script combines analytic calculations, rate-and-weight plasticity simulations, and precomputed convergence-basin grids in a 2 × 3 figure.

## Contents

- `Fig5.py`: figure analysis, simulations, and plotting.
- `Fig5data/`: precomputed convergence grids used in the bottom-row basin overlays.
- `Fig5.pdf`: generated output; created when the script is run.

## Required Python packages

- NumPy
- pandas
- Matplotlib
- scikit-learn

Install them with:

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

## Required data files

To create new heatmap data, run Fig5.py with `run_new=False`, `Fig5.py` expects these files:

```text
Fig5data/data_colorgrid_22.0.csv
Fig5data/data_colorgrid_10.0.csv
Fig5data/data_colorgrid_2.0.csv
```

The existing `run_new` switch can regenerate the CSV files using the simulations already contained in `Fig5.py`. This calculation is computationally expensive. No separate simulation script is required.

## Directory structure

```text
Figure5/
├── Fig5.py
├── README.md
└── Fig5data/
    ├── data_colorgrid_22.0.csv
    ├── data_colorgrid_10.0.csv
    └── data_colorgrid_2.0.csv
```

## Run

Run the script from this directory so that its relative data path resolves correctly:

```bash
python Fig5.py
```

The script writes `Fig5.pdf` to the current directory.


