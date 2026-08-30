# Figure 2, Row 3

This directory reproduces the **third row of Figure 2 in the current paper**. It contains only the multi-pulse learning workflow that in the third row of Figure 2.

## What is plotted

From left to right, `Figure2Row3.pdf` contains:

1. **Excitatory activity during learning:** the mean excitatory population rate, $\bar{r}_E(t)$, during the multi-pulse simulation. Line color indicates simulation time.
2. **Mean weights over time:** the normalized mean excitatory weight, $\bar{w}_{EE}/\bar{w}_{EE0}$, above the normalized mean inhibitory weight magnitude, $\bar{w}_{EI}/\bar{w}_{EI0}$. Line color indicates simulation time.
3. **Weight-space trajectory during learning:** the trajectory of the mean $(w_{EE}, w_{EI})$ pair. The plasma gradient indicates simulation time, and the dashed red curve is loaded from the original line-attractor reference files.

These are the learning panels now used as the third row of Figure 2 in the new paper.

## Files

- `RunFigure2Row3Simulation.py`: sets the original multi-pulse parameters, runs the simulation, and writes the CSV files.
- `Figure2Row3SpikingSim.py`: contains the original spiking dynamics and plasticity updates used by the runner.
- `Figure2Row3HelperFunctions.py`: contains only the weight-matrix helper functions referenced by the spiking simulation.
- `PlotFigure2Row3.py`: loads the saved simulation output and plots only the three requested panels.
- `Figure2Row3Data/`: destination for generated CSV files.

The old single-pulse runner, pattern-completion runner, and spiking-inset plotting file are not needed for these panels and are therefore not included.

## Required reference files
Coordinates for the simulated line attractor can be found in the following text files. If you wish to rerun and obtain these coordinates yourself, you just need to resimulate all of the initial conditions in the asynchronous state after some burnin. These text files are just to plot the line attractor for reference to see the drift along the line while perturbing the system. 

- `final_wEE_list_simulation.txt`
- `final_wEI_list_simulation.txt`

  
## Python requirements

The scripts use:

```text
numpy
pandas
matplotlib
```

## Usage

Run the commands from this directory:

```bash
python RunFigure2Row3Simulation.py
python PlotFigure2Row3.py
```

The simulation writes its outputs to `Figure2Row3Data/`. The plotting script then creates:

```text
Figure2Row3.pdf
```

The full simulation retains the original size and resolution: 500 excitatory neurons, 500 inhibitory neurons, 210 seconds of simulated time, and a 0.2 ms time step. It can therefore take substantial time and memory to run.

## Simulation files read by the plotting script

The plotting script uses the following outputs for seed 0:

- `multi_pulse_mean_rates_...csv`
- `multi_pulse_mean_ee_weights_...csv`
- `multi_pulse_mean_ei_weights_...csv`

The runner also retains the original saves for the tracked individual weights and balance index, although those two files are not plotted in this row.
