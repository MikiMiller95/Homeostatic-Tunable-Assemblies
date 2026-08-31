import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

try:
    from scipy.integrate import simpson
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from scipy.optimize import least_squares
    HAVE_SCIPY_OPTIMIZE = True
except Exception:
    HAVE_SCIPY_OPTIMIZE = False

N = 500
ms_per_sec = 1000.0
tau_rprim = 2.0 / ms_per_sec
scale_factor = np.sqrt(N)

def sc(raw):
    return raw / scale_factor * tau_rprim

w_IE = sc(10.0)
w_II = sc(10.0)
w_EX = sc(20.0)
w_IX = sc(15.0)
W_x = np.array([[w_EX, 0.0], [0.0, w_IX]], dtype=float)

tau_STDP = 120.0 / ms_per_sec
tau_r = 8.0 / ms_per_sec
tau_ou = 8.0 / ms_per_sec

aE = 20.0
aI = 20.0
b_default = 20.0
sigma_default = 1.5

GRID_N = 1001
FLOW_N = 10
w_EE_raw = np.linspace(0.0, 37.0, GRID_N)
w_EI_raw = np.linspace(0.0, 45.0, GRID_N)
w_EE_vals = sc(w_EE_raw)
w_EI_vals = sc(w_EI_raw)
w_ee, w_ei = np.meshgrid(w_EE_vals, w_EI_vals, indexing="xy")

freq = np.linspace(-2500.0, 2500.0, 10000)
Nw = freq.size

tau_ri = 2.0 * tau_r
gE_plus = 1.0 / (1.0 + 1j * 2.0 * np.pi * tau_r * freq)
gI_plus = 1.0 / (1.0 + 1j * 2.0 * np.pi * tau_ri * freq)
gE_minus = 1.0 / (1.0 - 1j * 2.0 * np.pi * tau_r * freq)
gI_minus = 1.0 / (1.0 - 1j * 2.0 * np.pi * tau_ri * freq)
ktilde = tau_STDP / (1.0 + 1j * 2.0 * np.pi * tau_STDP * freq)

NULLCLINE_COLOR_EE = "deepskyblue"
NULLCLINE_COLOR_EI = "tab:orange"
STAB_COLOR = "k"
POINT_COLOR = "red"
FLOW_COLOR = "k"

OUT_4PANEL_PNG = "data/homeostatic_compliance_nullclines_boundaries_flowfield_4panel_v8.png"
OUT_4PANEL_PDF = "data/homeostatic_compliance_nullclines_boundaries_flowfield_4panel_v8.pdf"
OUT_EONLY_PNG = "data/homeostatic_compliance_nullclines_boundaries_flowfield_Eonly_v8.png"
OUT_EONLY_PDF = "data/homeostatic_compliance_nullclines_boundaries_flowfield_Eonly_v8.pdf"
OUT_SUMMARY = "data/homeostatic_compliance_nullclines_boundaries_flowfield_v8_summary.txt"


def integrate_complex(y, x):
    if HAVE_SCIPY:
        return simpson(y, x=x, axis=-1)
    return np.trapz(y, x=x, axis=-1)


def external_covariance_matrix(sigma_E, sigma_I, sigma_cross, freq):
    ou_factor = 1.0 / (1.0 + (2.0 * np.pi * tau_ou * freq) ** 2)
    cov_matrix = np.zeros((2, 2, freq.size), dtype=float)
    cov_matrix[0, 0, :] = sigma_E**2 * ou_factor
    cov_matrix[1, 1, :] = sigma_I**2 * ou_factor
    cov_matrix[0, 1, :] = sigma_cross * ou_factor
    cov_matrix[1, 0, :] = sigma_cross * ou_factor
    cov_matrix *= N * (N - 1)
    cov_matrix[0, 0, :] += aE * N
    cov_matrix[1, 1, :] += aI * N
    return cov_matrix


def build_condition_cache(sigma_E, sigma_I, sigma_cross=0.0):
    lambda_cov = external_covariance_matrix(sigma_E, sigma_I, sigma_cross, freq)
    left_wx = W_x[:, :, None].astype(np.complex128) * gE_plus[None, None, :]
    right_wx = W_x.T[:, :, None].astype(np.complex128) * gE_minus[None, None, :]
    right_Xcov = np.einsum("ijk,jlk->ilk", lambda_cov.astype(np.complex128), left_wx)
    X_cov = np.einsum("ijk,jlk->ilk", right_wx, right_Xcov)
    return {"X_cov": X_cov}


def explicit_rates(wEE, wEI):
    det_M = (1.0 - N * wEE) * (1.0 + N * w_II) + (N**2) * wEI * w_IE
    if np.isclose(det_M, 0.0):
        return np.nan, np.nan, det_M
    rE = N * ((1.0 + N * w_II) * w_EX * aE - N * wEI * w_IX * aI) / det_M
    rI = N * (N * w_IE * w_EX * aE + (1.0 - N * wEE) * w_IX * aI) / det_M
    return rE, rI, det_M


def fixed_rates_and_stability(wEE, wEI):
    rE, rI, _ = explicit_rates(wEE, wEI)
    if not np.isfinite(rE) or not np.isfinite(rI):
        return np.nan, np.nan, np.nan
    W = np.array([[wEE, -wEI], [w_IE, -w_II]], dtype=float)
    tau_vec = np.array([tau_r, 2.0 * tau_r], dtype=float)
    J = np.diag(1.0 / tau_vec) @ (-np.eye(2) + N * W)
    vals = np.linalg.eigvals(J)
    stab = np.max(np.real(vals))
    return rE, rI, stab


def calculate_integrated_csd_for_point(W, lamb_e, lamb_i, cache):
    Full_W = np.repeat(W[:, :, None].astype(np.complex128), Nw, axis=2)
    Full_W[:, 0, :] *= gE_plus[None, :]
    Full_W[:, 1, :] *= gI_plus[None, :]

    Full_WT = np.repeat(W.T[:, :, None].astype(np.complex128), Nw, axis=2)
    Full_WT[0, :, :] *= gE_minus[None, :]
    Full_WT[1, :, :] *= gI_minus[None, :]

    a = 1.0 - Full_W[0, 0, :]
    b = -Full_W[0, 1, :]
    c = -Full_W[1, 0, :]
    d = 1.0 - Full_W[1, 1, :]
    det_left = a * d - b * c

    aT = 1.0 - Full_WT[0, 0, :]
    bT = -Full_WT[0, 1, :]
    cT = -Full_WT[1, 0, :]
    dT = 1.0 - Full_WT[1, 1, :]
    det_right = aT * dT - bT * cT

    if np.min(np.abs(det_left)) < 1e-8 or np.min(np.abs(det_right)) < 1e-8:
        return np.full((2, 2), np.nan, dtype=np.complex128)

    inv_term_1 = np.empty((2, 2, Nw), dtype=np.complex128)
    inv_term_1[0, 0, :] = d / det_left
    inv_term_1[0, 1, :] = -b / det_left
    inv_term_1[1, 0, :] = -c / det_left
    inv_term_1[1, 1, :] = a / det_left

    inv_term_2 = np.empty((2, 2, Nw), dtype=np.complex128)
    inv_term_2[0, 0, :] = dT / det_right
    inv_term_2[0, 1, :] = -bT / det_right
    inv_term_2[1, 0, :] = -cT / det_right
    inv_term_2[1, 1, :] = aT / det_right

    D = np.zeros((2, 2, Nw), dtype=np.complex128)
    D[0, 0, :] = lamb_e
    D[1, 1, :] = lamb_i

    right_Ktilde = np.einsum("ijk,jlk->ilk", D / (N - 1.0) + cache["X_cov"], inv_term_2)
    spectrum = np.einsum("ijk,jlk->ilk", inv_term_1, right_Ktilde)
    integrand = ktilde[None, None, :] * spectrum
    return integrate_complex(integrand, freq)


def analytic_positive_rate_boundaries():
    wEE_rI_zero = (aE * N * w_EX * w_IE + aI * w_IX) / (aI * N * w_IX)
    wEI_rE_zero = (aE * w_EX * (1.0 + N * w_II)) / (aI * N * w_IX)
    return wEE_rI_zero, wEI_rE_zero


def stationary_components(wEE, wEI, b_EE, b_EI, cache, require_physical=True):
    rE, rI, stab = fixed_rates_and_stability(wEE, wEI)
    if not np.isfinite(rE) or not np.isfinite(rI) or not np.isfinite(stab):
        return np.nan, np.nan, rE, rI, stab, np.nan, np.nan
    if require_physical and (rE <= 0.0 or rI <= 0.0 or stab >= -1e-3):
        return np.nan, np.nan, rE, rI, stab, np.nan, np.nan
    W = N * np.array([[wEE, -wEI], [w_IE, -w_II]], dtype=float)
    csd_int = calculate_integrated_csd_for_point(W, rE, rI, cache)
    if np.any(~np.isfinite(csd_int)):
        return np.nan, np.nan, rE, rI, stab, np.nan, np.nan
    C_EE = np.real(csd_int[0, 0])
    C_IE = np.real(csd_int[1, 0])
    tau_w_e = 225000  # 63000 s
    tau_w_i = 75000   # 21000 s
    #dot_wEE = tau_STDP * rE * (rE - b_EE) + C_EE
    #dot_wEI = tau_STDP * rI * (rE - b_EI) + C_IE
    dot_wEE = (tau_STDP * rE * (rE - b_EE) + C_EE) / tau_w_e
    dot_wEI = (tau_STDP * rI * (rE - b_EI) + C_IE) / tau_w_i
    return dot_wEE, dot_wEI, rE, rI, stab, C_EE, C_IE


def is_async_compliant_line_attractor(condition):
    return np.isclose(condition["b_EE"], condition["b_EI"]) and np.isclose(condition["sigma_E"], 0.0) and np.isclose(condition["sigma_I"], 0.0) and np.isclose(condition["sigma_cross"], 0.0)


def find_stationary_points(condition, cache, seed_residual_grid, require_physical=True, top_k=8, coarse_n=5, tol=1e-6):
    if not HAVE_SCIPY_OPTIMIZE or not np.any(np.isfinite(seed_residual_grid)):
        return []
    bounds = ([w_EE_vals.min(), w_EI_vals.min()], [w_EE_vals.max(), w_EI_vals.max()])

    def objective(x):
        f1, f2, _, _, _, _, _ = stationary_components(x[0], x[1], condition["b_EE"], condition["b_EI"], cache, require_physical=require_physical)
        if not np.isfinite(f1) or not np.isfinite(f2):
            return np.array([1e3, 1e3], dtype=float)
        return np.array([f1, f2], dtype=float)

    seeds = []
    finite_mask = np.isfinite(seed_residual_grid)
    finite_idx = np.flatnonzero(finite_mask)
    if finite_idx.size > 0:
        order = np.argsort(seed_residual_grid[finite_mask])
        for idx in finite_idx[order[:min(top_k, order.size)]]:
            j, i = np.unravel_index(idx, seed_residual_grid.shape)
            seeds.append(np.array([w_EE_vals[i], w_EI_vals[j]], dtype=float))
    for wEE0 in np.linspace(w_EE_vals.min(), w_EE_vals.max(), coarse_n):
        for wEI0 in np.linspace(w_EI_vals.min(), w_EI_vals.max(), coarse_n):
            seeds.append(np.array([wEE0, wEI0], dtype=float))

    roots = []
    for x0 in seeds:
        sol = least_squares(objective, x0, bounds=bounds, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=70)
        res = objective(sol.x)
        if np.linalg.norm(res) > tol:
            continue
        f1, f2, rE_star, rI_star, stab_star, C_EE_star, C_IE_star = stationary_components(sol.x[0], sol.x[1], condition["b_EE"], condition["b_EI"], cache, require_physical=require_physical)
        if not np.isfinite(rE_star) or not np.isfinite(rI_star):
            continue
        if any(np.linalg.norm(sol.x - np.array([rt["wEE"], rt["wEI"]])) < 1e-7 for rt in roots):
            continue
        roots.append({
            "wEE": sol.x[0], "wEI": sol.x[1], "rE": rE_star, "rI": rI_star, "stab": stab_star,
            "C_EE": C_EE_star, "C_IE": C_IE_star, "dot_wEE": f1, "dot_wEI": f2, "res_norm": np.linalg.norm([f1, f2])
        })
    roots.sort(key=lambda rt: (rt["wEE"], rt["wEI"]))
    return roots


def compute_panel(condition, require_physical=True):
    cache = build_condition_cache(condition["sigma_E"], condition["sigma_I"], condition["sigma_cross"])
    dot_wEE_grid = np.full((len(w_EI_vals), len(w_EE_vals)), np.nan)
    dot_wEI_grid = np.full_like(dot_wEE_grid, np.nan)
    rE_grid = np.full_like(dot_wEE_grid, np.nan)
    rI_grid = np.full_like(dot_wEE_grid, np.nan)
    stab_grid = np.full_like(dot_wEE_grid, np.nan)

    for j, wEI in enumerate(w_EI_vals):
        for i, wEE in enumerate(w_EE_vals):
            rE, rI, stab = fixed_rates_and_stability(wEE, wEI)
            rE_grid[j, i] = rE
            rI_grid[j, i] = rI
            stab_grid[j, i] = stab
            f1, f2, _, _, _, _, _ = stationary_components(wEE, wEI, condition["b_EE"], condition["b_EI"], cache, require_physical=require_physical)
            dot_wEE_grid[j, i] = f1
            dot_wEI_grid[j, i] = f2

    residual_grid = np.sqrt(dot_wEE_grid**2 + dot_wEI_grid**2)
    if require_physical and is_async_compliant_line_attractor(condition):
        roots = []
    else:
        roots = find_stationary_points(condition, cache, residual_grid, require_physical=require_physical)

    min_residual = np.nan
    min_point = None
    finite = np.isfinite(residual_grid)
    if np.any(finite):
        j0, i0 = np.unravel_index(np.nanargmin(residual_grid), residual_grid.shape)
        min_residual = residual_grid[j0, i0]
        min_point = (w_EE_vals[i0], w_EI_vals[j0], rE_grid[j0, i0], rI_grid[j0, i0])

    return {
        "dot_wEE": dot_wEE_grid, "dot_wEI": dot_wEI_grid, "rE": rE_grid, "rI": rI_grid,
        "stab": stab_grid, "residual": residual_grid, "roots": roots, "min_residual": min_residual,
        "min_point": min_point, "cache": cache
    }


def add_flow_field(ax, result):
    idx_x = np.linspace(0, len(w_EE_vals) - 1, FLOW_N, dtype=int)
    idx_y = np.linspace(0, len(w_EI_vals) - 1, FLOW_N, dtype=int)
    Xq, Yq = np.meshgrid(w_EE_vals[idx_x], w_EI_vals[idx_y], indexing="xy")
    U = result["dot_wEE"][np.ix_(idx_y, idx_x)].astype(float)
    V = result["dot_wEI"][np.ix_(idx_y, idx_x)].astype(float)
    M = np.hypot(U, V)
    valid = np.isfinite(M) & (M > 0)
    Uplot = np.full_like(U, np.nan)
    Vplot = np.full_like(V, np.nan)
    Uplot[valid] = U[valid] / M[valid]
    Vplot[valid] = V[valid] / M[valid]
    ax.quiver(Xq, Yq, Uplot, Vplot, angles="xy", scale_units="xy", scale=6000.0, width=0.004, headwidth=3.8, headlength=5.0, headaxislength=4.8, pivot="mid", color=FLOW_COLOR, alpha=0.9, zorder=1)


def plot_condition_panel(ax, condition, result, add_legend=False):
    add_flow_field(ax, result)
    ax.axvline(wEE_rI_zero, color="k", linestyle=":", linewidth=1.5)
    ax.axhline(wEI_rE_zero, color="k", linestyle=":", linewidth=1.5)
    ax.contour(w_ee, w_ei, result["stab"], levels=[0.0], colors=STAB_COLOR, linewidths=1.8, linestyles="--", zorder=2)
    ax.contour(w_ee, w_ei, result["dot_wEE"], levels=[0.0], colors=NULLCLINE_COLOR_EE, linewidths=2.3, zorder=3)
    ax.contour(w_ee, w_ei, result["dot_wEI"], levels=[0.0], colors=NULLCLINE_COLOR_EI, linewidths=2.3, zorder=3)

    for root in result["roots"]:
        ax.scatter([root["wEE"]], [root["wEI"]], s=58, color=POINT_COLOR, edgecolors="w", linewidths=0.7, zorder=6)

    if add_legend:
        ax.plot([], [], color=FLOW_COLOR, linewidth=1.8)
        ax.plot([], [], color=STAB_COLOR, linestyle="--", linewidth=1.8, label="rate stability boundary")
        ax.plot([], [], color=NULLCLINE_COLOR_EE, linewidth=2.3, label=r"$\dot{w}_{EE}=0$")
        ax.plot([], [], color=NULLCLINE_COLOR_EI, linewidth=2.3, label=r"$\dot{w}_{EI}=0$")
        ax.scatter([], [], s=58, color=POINT_COLOR, edgecolors="w", linewidths=0.7, label="Fixed Point")
        ax.legend(frameon=False, fontsize=9, loc="center left")

    ax.set_title(condition["title"])
    ax.set_xlim(w_EE_vals.min(), w_EE_vals.max())
    ax.set_ylim(w_EI_vals.min(), w_EI_vals.max())
    ax.set_xlabel(r"$w_{EE}$")
    ax.set_ylabel(r"$w_{EI}$")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    global wEE_rI_zero, wEI_rE_zero
    conditions = [
        {"title": "Asynchronous and compliant", "b_EE": b_default, "b_EI": b_default, "sigma_E": 0.0, "sigma_I": 0.0, "sigma_cross": 0.0},
        {"title": "Asynchronous and noncompliant", "b_EE": 30.0, "b_EI": 20.0, "sigma_E": 0.0, "sigma_I": 0.0, "sigma_cross": 0.0},
        {"title": "Correlated and compliant", "b_EE": b_default, "b_EI": b_default, "sigma_E": sigma_default, "sigma_I": sigma_default, "sigma_cross": 0.0},
        {"title": "Correlated and noncompliant", "b_EE": 30.0, "b_EI": 20.0, "sigma_E": sigma_default, "sigma_I": sigma_default, "sigma_cross": 0.0},
    ]

    results = [compute_panel(cond, require_physical=True) for cond in conditions]
    wEE_rI_zero, wEI_rE_zero = analytic_positive_rate_boundaries()
    case3_any_roots = find_stationary_points(conditions[2], results[2]["cache"], results[2]["residual"], require_physical=False)

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.2), constrained_layout=True)
    for idx, (ax, cond, res) in enumerate(zip(axes, conditions, results)):
        plot_condition_panel(ax, cond, res, add_legend=(idx == 0))
    fig.suptitle("Plasticity nullclines, flow fields, and dynamical boundaries", fontsize=14)
    fig.savefig(OUT_4PANEL_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_4PANEL_PDF, bbox_inches="tight")
    plt.close(fig)

    condition_E_only = {"title": "E-only correlated and compliant", "b_EE": b_default, "b_EI": b_default, "sigma_E": sigma_default, "sigma_I": 0.0, "sigma_cross": 0.0}
    result_E_only = compute_panel(condition_E_only, require_physical=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(5.2, 5.2), constrained_layout=True)
    plot_condition_panel(ax2, condition_E_only, result_E_only, add_legend=True)
    fig2.savefig(OUT_EONLY_PNG, dpi=300, bbox_inches="tight")
    fig2.savefig(OUT_EONLY_PDF, bbox_inches="tight")
    plt.close(fig2)

    with open(OUT_SUMMARY, "w") as f:
        f.write("Flow-field version. Nullcline and fixed-point analysis unchanged.\n\n")
        f.write("Explicit rate expressions used\n")
        f.write("  det M = (1 - N w_EE)(1 + N w_II) + N^2 w_EI w_IE\n")
        f.write("  r_E^*(w) = N[(1 + N w_II) w_EX a_E - N w_EI w_IX a_I] / det M\n")
        f.write("  r_I^*(w) = N[N w_IE w_EX a_E + (1 - N w_EE) w_IX a_I] / det M\n\n")
        f.write("Covariance terms used\n")
        f.write("  C_EE(w), C_IE(w) = Re ∫ dω K~(ω) [(I - W~(ω))^{-1}(D/(N-1)+X_cov)(I - W~(-ω)^T)^{-1}]_{•E},\n")
        f.write("  with D = diag(r_E^*(w), r_I^*(w)).\n\n")
        f.write("Four main panels\n")
        for cond, res in zip(conditions, results):
            f.write(f"- {cond['title']}\n")
            if is_async_compliant_line_attractor(cond):
                f.write("  non-isolated stationary manifold (line attractor), so no single isolated fixed point\n")
            elif len(res["roots"]) == 0:
                f.write("  no stationary point found in the positive-rate stable plotted region\n")
            else:
                for root in res["roots"]:
                    f.write(f"  isolated fixed point: wEE={root['wEE']:.12g}, wEI={root['wEI']:.12g}, rE={root['rE']:.12g}, rI={root['rI']:.12g}, C_EE={root['C_EE']:.12g}, C_IE={root['C_IE']:.12g}\n")
            if np.isfinite(res['min_residual']):
                f.write(f"  min residual in positive/stable plotted region: {res['min_residual']:.12g}\n")
            f.write("\n")
        f.write("Correlated and compliant: unrestricted search in the plotted weight box\n")
        if len(case3_any_roots) == 0:
            f.write("- no stationary point found anywhere in the plotted box, even without the positive-rate/stability restriction\n\n")
        else:
            for root in case3_any_roots:
                f.write(f"- unrestricted stationary point: wEE={root['wEE']:.12g}, wEI={root['wEI']:.12g}, rE={root['rE']:.12g}, rI={root['rI']:.12g}, C_EE={root['C_EE']:.12g}, C_IE={root['C_IE']:.12g}, stab={root['stab']:.12g}\n")
            f.write("\n")
        f.write("E-only correlated and compliant (sigma_E = 0.5, sigma_I = 0)\n")
        if len(result_E_only["roots"]) == 0:
            f.write("- no stationary point found in the positive-rate stable plotted region\n")
        else:
            for root in result_E_only["roots"]:
                f.write(f"- isolated fixed point: wEE={root['wEE']:.12g}, wEI={root['wEI']:.12g}, rE={root['rE']:.12g}, rI={root['rI']:.12g}, C_EE={root['C_EE']:.12g}, C_IE={root['C_IE']:.12g}\n")
        f.write("- note: if only E receives correlated external fluctuations, C_EE and C_IE are generally not zero, because recurrent coupling propagates the E-input fluctuations through the network.\n")

    print("Wrote:")
    for path in [OUT_4PANEL_PNG, OUT_4PANEL_PDF, OUT_EONLY_PNG, OUT_EONLY_PDF, OUT_SUMMARY]:
        print(path)


if __name__ == "__main__":
    main()
