> The dataset is available at [Kaggle — MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) or the [official MNIST page](http://yann.lecun.com/exdb/mnist/).

---

## Requirements

Python 3.8+ is recommended.

### Install dependencies

```bash
pip install numpy matplotlib scikit-learn
```

### Full dependency list

| Package | Version (tested) | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Matrix operations, eigendecomposition |
| `matplotlib` | ≥ 3.4 | Visualizations (sample images, scatter plots) |
| `scikit-learn` | ≥ 0.24 | `mean_squared_error` for reconstruction evaluation only |

> **Note:** scikit-learn is used **only** for MSE computation. All classifiers and dimensionality reduction methods are custom-built.

---

## Pipeline

### Step 1 — Load Data
Parse raw IDX binary files into NumPy arrays.

### Step 2 — Sample & Balance
Randomly sample 100 images per class (digits 0, 1, 2) for both train and test sets.

### Step 3 — Vectorize & Normalize
Flatten each 28×28 image to a 784-d vector (column-major order) and scale to `[0, 1]`.

### Step 4 — PCA (Custom)
- Compute covariance matrix on centered training data
- Eigendecomposition via `np.linalg.eigh`
- Support for variance-threshold mode (`keep_variance`) and fixed-k mode (`keep_components`)
- Includes `reconstruct()` for image recovery and MSE evaluation

**Reconstruction MSE (75% variance, 19 components):** `0.01185`

### Step 5 — FDA (Custom)
- Compute within-class scatter (S_W) and between-class scatter (S_B)
- Solve generalized eigenvalue problem on `pinv(S_W + reg) @ S_B`
- Projects to `C-1 = 2` discriminant dimensions for 3 classes

### Step 6 — Classification
- **LDA** — shared covariance Gaussian classifier (linear decision boundaries)
- **QDA** — per-class covariance Gaussian classifier (quadratic decision boundaries)
- Both use log-likelihood scoring with Gaussian priors

### Step 7 — Evaluation
Train and test accuracy reported for each method combination.

---

## Results

| Method | Train Accuracy | Test Accuracy |
|---|---|---|
| FDA + LDA | 100.00% | 85.33% |
| FDA + QDA | 100.00% | 85.33% |
| PCA (75%, 19 PCs) + LDA | 97.00% | 94.67% |
| **PCA (90%, 51 PCs) + LDA** | **99.67%** | **96.00%** ✅ |
| PCA (2 PCs) + LDA | 92.67% | 91.00% |

**Best model:** PCA (90% variance threshold) + LDA achieves **96% test accuracy** with 51 principal components.

### Observations

- FDA reduces to just 2 dimensions (hard limit of C-1 for C=3 classes), causing an information bottleneck versus PCA at higher variance thresholds.
- LDA and QDA perform identically on FDA features — the 2D projection likely makes class boundaries approximately linear, removing any advantage of per-class covariances.
- Increasing PCA variance from 75% → 90% (19 → 51 components) yields a meaningful +1.33% test accuracy gain.
- Even 2-component PCA achieves 91% test accuracy, showing digits 0/1/2 are well-separated in PC space.

---

## Key Design Choices

- **Column-major flattening (`order='F'`)** when vectorizing and reshaping images, matching the PCA training convention.
- **Regularization (`1e-6 * I`)** added to covariance matrices in FDA and classifiers to prevent singularity on the small 300-sample dataset.
- **`np.linalg.eigh`** (symmetric eigenvalue solver) used in PCA for numerical stability over the general `eig` solver.
- **Balanced sampling** ensures no class imbalance affects accuracy metrics.
