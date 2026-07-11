# AI R&D Assignment — Parametric Curve Fitting

## Problem

Find `theta`, `M`, `X` in:

```
x(t) = t*cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta) + X
y(t) = 42 + t*sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta)
```

for `6 < t < 60`, given 1500 `(x, y)` points sampled from the curve (`xy_data.csv`), subject to:

```
0 deg < theta < 50 deg
-0.05 < M < 0.05
0 < X < 100
```

## Approach

The rows in `xy_data.csv` have no accompanying `t` value and are **not** given in
order of `t`, so this can't be solved as an ordinary regression of `x_i -> y_i`
against a known `t_i`. It's a **point-cloud-to-curve fitting** problem instead:

1. **Candidate curve generation.** For a candidate `(theta, M, X)`, densely
   sample the curve over `t ∈ [6, 60]` to build a fine polyline approximating
   that candidate curve.
2. **Nearest-point cost.** For every data point, find the distance to the
   *closest* point on that candidate curve (no assumed correspondence/order
   between data rows and `t`). Sum the squared distances — this is the cost
   function to minimize.
3. **Global search.** Because the "nearest point on a polyline" operation
   makes the cost surface non-smooth/non-convex, use
   `scipy.optimize.differential_evolution` over the given parameter bounds
   to find the global basin.
4. **Local polish.** Refine that result with `scipy.optimize.minimize`
   (Nelder-Mead) on a much finer `t`-grid (5000 points) for high precision.

## Result

The optimizer converges to clean, "round" values with near-zero residual:

| Parameter | Value            |
|-----------|------------------|
| `theta`   | 30°  (0.5235987756 rad) |
| `M`       | 0.03             |
| `X`       | 55               |

**Fit quality:** max nearest-point residual ≈ 0.0023, mean ≈ 0.0008 (in data
units) — consistent with floating-point/rounding noise in how the dataset
was generated, not a genuine mismatch. This strongly indicates these are the
exact ground-truth parameters (θ=30°, M=0.03, X=55), and the optimizer simply
recovered them.

## Desmos equation (for verification)

```
\left(t*\cos(0.5235987756)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5235987756)+55,42+t*\sin(0.5235987756)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5235987756)\right)
```
Domain: `6 ≤ t ≤ 60`

## Files

- `fit_curve.py` — full solution: loads `xy_data.csv`, runs the two-stage
  optimization, prints the fitted parameters, fit-quality stats, and the
  ready-to-paste Desmos LaTeX string.
- `xy_data.csv` — provided dataset (included for reproducibility).
- `requirements.txt` — dependencies.

## Usage

```bash
pip install -r requirements.txt
python fit_curve.py
```
