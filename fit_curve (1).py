"""
AI R&D Assignment - Parametric Curve Fitting
==============================================

Problem
-------
Find (theta, M, X) such that the parametric curve

    x(t) = t*cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta)

    for 6 < t < 60

passes through (i.e. is the source of) the points given in xy_data.csv.

Approach
--------
The rows in xy_data.csv are NOT given in order of the parameter t (there is
no column identifying which t produced which point), so this cannot be
solved as a simple curve_fit(x_i -> y_i) regression against a known t_i.

Instead this is treated as a point-cloud-to-curve fitting problem:

1. For a candidate (theta, M, X), densely sample the curve over t in
   [6, 60] to build a fine polyline approximation of the candidate curve.
2. For every data point, compute the distance to the *nearest* point on
   that candidate curve (i.e. we don't assume correspondence/order between
   data rows and t values).
3. Sum the squared nearest-point distances -> this is the cost function.
4. Minimize the cost over (theta, M, X) using:
      a) scipy.optimize.differential_evolution (global search, since the
         cost surface is not smooth/convex due to the "nearest point"
         operation), followed by
      b) scipy.optimize.minimize (Nelder-Mead) on a much finer t-grid to
         polish the global result to high precision.

Bounds used (from the assignment):
    0 deg  < theta < 50 deg
   -0.05   < M     < 0.05
    0      < X     < 100

Result
------
The optimizer converges to very clean, "round" values with near-zero
residual (max distance ~0.002, consistent with floating point / rounding
noise in how the data was generated), strongly suggesting these are the
exact ground-truth parameters:

    theta = 30 deg  = 0.5235987756 rad
    M     = 0.03
    X     = 55

Usage
-----
    pip install -r requirements.txt
    python fit_curve.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

DATA_PATH = "xy_data.csv"

# ----------------------------------------------------------------------
# Curve model
# ----------------------------------------------------------------------
def curve_xy(theta, M, X, t):
    """Evaluate the parametric curve at parameter value(s) t."""
    ct, st = np.cos(theta), np.sin(theta)
    e = np.exp(M * np.abs(t))
    s3 = np.sin(0.3 * t)
    x = t * ct - e * s3 * st + X
    y = 42 + t * st + e * s3 * ct
    return x, y


# ----------------------------------------------------------------------
# Cost function: sum of squared nearest-point distances
# ----------------------------------------------------------------------
def make_cost(x_data, y_data, t_grid):
    def cost(params):
        theta, M, X = params
        cx, cy = curve_xy(theta, M, X, t_grid)
        dx = x_data[:, None] - cx[None, :]
        dy = y_data[:, None] - cy[None, :]
        d2 = dx ** 2 + dy ** 2
        return d2.min(axis=1).sum()
    return cost


def fit(x_data, y_data):
    bounds = [
        (1e-4, np.deg2rad(50) - 1e-4),   # theta (radians)
        (-0.05, 0.05),                   # M
        (1e-4, 100 - 1e-4),              # X
    ]

    # Stage 1: coarse global search
    t_coarse = np.linspace(6, 60, 800)
    cost_coarse = make_cost(x_data, y_data, t_coarse)
    de_result = differential_evolution(
        cost_coarse, bounds, maxiter=60, popsize=15, tol=1e-8, seed=1, polish=True
    )

    # Stage 2: local polish on a much finer t-grid
    t_fine = np.linspace(6, 60, 5000)
    cost_fine = make_cost(x_data, y_data, t_fine)
    result = minimize(
        cost_fine, de_result.x, method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000},
    )

    return result.x, result.fun


def report(theta, M, X, x_data, y_data):
    t_dense = np.linspace(6, 60, 20000)
    cx, cy = curve_xy(theta, M, X, t_dense)
    dx = x_data[:, None] - cx[None, :]
    dy = y_data[:, None] - cy[None, :]
    dist = np.sqrt((dx ** 2 + dy ** 2).min(axis=1))

    print("\n========== Estimated Parameters ==========")
    print(f"theta = {theta:.10f} rad  ({np.rad2deg(theta):.6f} deg)")
    print(f"M     = {M:.10f}")
    print(f"X     = {X:.10f}")
    print("\n========== Fit Statistics ==========")
    print(f"max residual  : {dist.max():.6f}")
    print(f"mean residual : {dist.mean():.6f}")

    print("\n========== Desmos-Compatible Equation ==========")
    print(
        f"\\left(t*\\cos({theta:.10f})-e^{{{M:.10f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\sin({theta:.10f})+{X:.10f},"
        f"42+t*\\sin({theta:.10f})+e^{{{M:.10f}\\left|t\\right|}}"
        f"\\cdot\\sin(0.3t)\\cos({theta:.10f})\\right)"
    )
    print("Domain: 6 <= t <= 60")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    x_data = df["x"].values
    y_data = df["y"].values

    (theta, M, X), final_cost = fit(x_data, y_data)
    report(theta, M, X, x_data, y_data)
