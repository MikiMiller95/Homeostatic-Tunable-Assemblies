[README(10).md](https://github.com/user-attachments/files/31623633/README.10.md)
# Figure 6: Homeostatic-compliance nullclines, boundaries, and flow fields

This directory contains the analysis used to generate the **last row of panels in Figure 6**. The script computes the plasticity flow directly in the \((w_{EE}, w_{EI})\) plane and compares how homeostatic compliance and correlated external fluctuations change the geometry of the synaptic-weight dynamics.

The script does **not** load previously generated simulation data. It calculates the fixed rates, covariance-dependent plasticity terms, nullclines, rate-stability boundary, flow field, and stationary points directly from the two-population theory.

## Files

- `homeostatic_compliance_nullclines_boundaries_flowfield_v8.py` — computes the theory over the synaptic-weight grid and writes the panel figures and a text summary.
- `README.md` — documents the purpose, execution order, outputs, and current implementation details.
- `data/` — output directory. This directory must exist before the script is run.

## Figure 6 panel organization

The publication-relevant output for the last row of Figure 6 is:

```text
data/homeostatic_compliance_nullclines_boundaries_flowfield_4panel_v8.pdf
```

The four panels are ordered from left to right as follows:

1. **Asynchronous and compliant**: \(b_{EE}=20\), \(b_{EI}=20\), and no correlated external fluctuations.
2. **Asynchronous and noncompliant**: \(b_{EE}=30\), \(b_{EI}=20\), and no correlated external fluctuations.
3. **Correlated and compliant**: \(b_{EE}=20\), \(b_{EI}=20\), \(\sigma_E=\sigma_I=1.5\), and \(\sigma_{\mathrm{cross}}=0\).
4. **Correlated and noncompliant**: \(b_{EE}=30\), \(b_{EI}=20\), \(\sigma_E=\sigma_I=1.5\), and \(\sigma_{\mathrm{cross}}=0\).

Here, **compliant** means that the excitatory and inhibitory plasticity rules use the same postsynaptic target, \(b_{EE}=b_{EI}\). **Noncompliant** means that the two targets differ. In the panels labeled “correlated,” the E and I external fluctuation variances are nonzero, but the explicit E–I cross-covariance remains zero.

The script also generates a separate E-only correlated panel. This is an additional analysis output and is not part of the four-panel row unless it is intentionally incorporated elsewhere in the figure.

## What is plotted

Each panel shows the following quantities in the \((w_{EE}, w_{EI})\) plane:

- **Black arrows**: direction of the synaptic-weight flow \((\dot{w}_{EE}, \dot{w}_{EI})\). The arrows are normalized, so they show direction rather than flow magnitude.
- **Cyan contour**: the excitatory-plasticity nullcline, \(\dot{w}_{EE}=0\).
- **Orange contour**: the inhibitory-plasticity nullcline, \(\dot{w}_{EI}=0\).
- **Black dashed contour**: the rate-stability boundary, defined by the largest real part of the rate Jacobian eigenvalues crossing zero.
- **Black dotted vertical and horizontal lines**: the analytic \(r_I^*=0\) and \(r_E^*=0\) boundaries.
- **Red points**: isolated stationary points found where both plasticity components vanish within the positive-rate, stable region.

For the asynchronous compliant condition, the script treats the stationary set as a non-isolated line attractor rather than marking a single fixed point.

## Analysis performed by the script

For every point on the synaptic-weight grid, the script:

1. Computes the explicit fixed-point excitatory and inhibitory rates.
2. Evaluates the local linear rate stability.
3. Rejects points outside the positive-rate, stable region when constructing the physical nullclines.
4. Computes the frequency-domain covariance contribution to plasticity.
5. Evaluates

   ```text
   dot_wEE = [tau_STDP rE (rE - b_EE) + C_EE] / tau_w_e
   dot_wEI = [tau_STDP rI (rE - b_EI) + C_IE] / tau_w_i
   ```

6. Extracts the two plasticity nullclines and the rate-stability boundary.
7. Uses bounded least-squares searches to locate isolated stationary points when appropriate.
8. Writes a text summary containing the rate equations, covariance expression, stationary-point results, and minimum residuals.

## Main parameter values

The script currently uses:

```text
N = 500
tau_rprim = 2 ms
tau_r = 8 ms
tau_I = 2 tau_r = 16 ms
tau_ou = 8 ms
tau_STDP = 120 ms
aE = 20
aI = 20
b_default = 20
sigma_default = 1.5
GRID_N = 1001
FLOW_N = 10
frequency range = -2500 to 2500 Hz
number of frequency samples = 10000
```

The synaptic weights are scaled through

```text
sc(raw) = raw / sqrt(N) * tau_rprim.
```

The literal plasticity timescale values used in the flow equations are `tau_w_e = 225000` and `tau_w_i = 75000`.

## Requirements

Use Python 3 with:

```text
numpy
matplotlib
scipy
```

Install the dependencies with:

```bash
python -m pip install numpy matplotlib scipy
```

SciPy is strongly recommended for reproduction. Without `scipy.integrate.simpson`, the script falls back to NumPy trapezoidal integration. Without `scipy.optimize.least_squares`, isolated stationary-point searches are skipped, so red fixed-point markers and the corresponding summary entries will be absent.

## Running the analysis

From this directory, first create the output folder:

```bash
mkdir -p data
```

Then run:

```bash
python homeostatic_compliance_nullclines_boundaries_flowfield_v8.py
```

The script uses Matplotlib's non-interactive `Agg` backend, so it can run on a server or compute node without opening a display window.

No other repository scripts or precomputed CSV files are required.

## Generated outputs

A successful run writes:

```text
data/homeostatic_compliance_nullclines_boundaries_flowfield_4panel_v8.png
data/homeostatic_compliance_nullclines_boundaries_flowfield_4panel_v8.pdf
data/homeostatic_compliance_nullclines_boundaries_flowfield_Eonly_v8.png
data/homeostatic_compliance_nullclines_boundaries_flowfield_Eonly_v8.pdf
data/homeostatic_compliance_nullclines_boundaries_flowfield_v8_summary.txt
```

### Primary Figure 6 output

- `homeostatic_compliance_nullclines_boundaries_flowfield_4panel_v8.pdf` is the vector-format row intended for incorporation into Figure 6.
- The matching PNG is useful for quick inspection.

### Additional outputs

- The `Eonly` PDF and PNG show the compliant condition when only the excitatory population receives correlated external fluctuations.
- The summary text file records the equations used and the stationary-point search results for all conditions.

