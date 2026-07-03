# Case Study: When Regression Fails (products-1000.csv)

This case study documents a real outcome from the AI Data Analyst app: predicting **Price** from product metadata in `sample-data/products-1000.csv`.

## The dataset

- ~1,000 rows of product catalog data
- Features: `Index`, `Name`, `Category`, `Brand`, `Color`, `Size`, `Material`, `Availability`, etc.
- Target: `Price` (continuous)

At first glance this looks like a standard regression problem. The app trains linear regression, random forest, and XGBoost automatically.

## What happened

**Best model R² was negative** (often around -0.1 to -0.3 depending on split and encoding).

A negative R² means the model performs **worse than predicting the mean price for every row**. That is not a bug — it is an honest signal that the available features do not explain price variation in this file.

## Why it fails

### 1. Weak feature–target relationship

`Category`, `Brand`, and `Color` have limited cardinality and weak correlation with price. There is no quantity, cost, margin, or market signal. Price appears effectively independent of the metadata columns provided.

### 2. Identifier and high-cardinality columns

`Name` and `Index` are near-unique per row. Using them as features causes overfitting on the training split and unstable generalization — classic **leakage-like behavior** without being true label leakage.

### 3. Encoding artifacts

Before fixes, frequency encoding on `Color` could dominate feature importance (~57%) because the encoder maps categories to a single numeric column whose variance does not generalize. The app now:

- Uses **target mean encoding** for high-cardinality categoricals in regression
- **Aggregates** one-hot and encoded importances back to source columns (e.g. `Color`, not `Color_Red`)

This makes explainability honest but does not magically improve R² when signal is absent.

### 4. Baseline comparison

R² = 0 is the mean predictor. Negative R² means the model adds noise. The ML page warning explains this explicitly.

## What we did in the app

| Improvement | Purpose |
|-------------|---------|
| Cross-validation (3-fold) | Stable metrics on small tabular data |
| Mean encoding (regression) | Better high-cardinality categoricals than raw frequency |
| Outlier treatment (winsorize/clip/remove) | Optional cleaning before training |
| Aggregated feature importance | Interpretable column-level rankings |
| Negative R² warning | User education, not hidden failure |
| Rich validation report | Flag identifier-like columns early |

## Recommended workflow for this file

1. **Overview** — read validation report; note identifier columns
2. **Do not expect Price prediction** — treat as exploration, not forecasting
3. **Clustering / Anomaly** — segment products or find unusual listings instead
4. **Chat** — ask *"Why is R² negative?"* — tools + RAG retrieve ML metrics and warnings
5. **Case study** — document the limitation in portfolio narrative (shows judgment, not just accuracy)

## Portfolio takeaway

> A senior data scientist knows when a model **should not** be deployed. This app surfaces failure modes transparently: negative R², CV variance, identifier warnings, and encoding-aware explainability.

That is more credible than claiming 90% accuracy on a problem with no predictive features.

## Try it yourself

```bash
docker compose up --build
# Upload sample-data/products-1000.csv
# Target: Price → Train → Read warnings on ML page
```

See also: [README](../README.md) · [Live demo](https://ai-data-analyst-app-sigma.vercel.app)
