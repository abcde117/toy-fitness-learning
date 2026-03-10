# Copilot Instructions for toy_fitness_learning

## Project Overview
This is a **score-based representation learning framework** exploring neural network encoding and decoding pipelines on wine quality data. It compares classical ML (Kernel Ridge Regression) with PyTorch/JAX-based approaches using graph convolutions and energy-based score fields.

**Core architecture**: Two-stage training pipeline—encoding (representation learning via kernel matching) followed by decoding (score field approximation via energy gradients).

---

## Architecture: The Big Picture

### Multi-Channel Graph Encoding
The encoder captures multiple relational patterns simultaneously via stacked adjacency matrices:

```python
g1 = build_type_graph(t)  # Binary connections: samples with same wine type
g2 = build_type_graph(q)  # Binary connections: samples with similar quality
g = torch.stack([g1, g2], dim=0)  # (2, N, N) — 2 relational channels
```

**Critical design**: `navie_nn` applies separate graph convolutions per channel, then aggregates learned representations using `aggloremation_layer` (learnable weighted sum). This multi-channel design is **essential**—preserve it when modifying encoding logic.

### Two-Stage Training Pipeline

#### Stage 1: Encoding (Representation Learning)
**File**: [nn_torch_ver/trainer.py](nn_torch_ver/trainer.py) → `exp_runner()`

- **Input**: Numeric features `x` + graph adjacency `g`
- **Model**: `navie_nn` (graph-aware encoder)
- **Process**:
  1. Graph convolution per channel via `navie_GCN(x, g)`
  2. Feature projection via `navie_fearue_head(x)`
  3. Multi-channel aggregation via `aggloremation_layer()`
- **Loss**: MIM loss (kernel matching) + Stat loss (feature statistics)
  - `MIM_loss = ||m @ m.T - k||²` where `m` is learned representation, `k` is kernel
  - `stat_loss = ||m_mean - x_mean||²` for feature-level statistics
- **Hyperparameters**: `lambda1` (MIM weight), `lambda2` (stat weight)
- **Output**: Frozen encoder for Stage 2

#### Stage 2: Decoding (Score Field Learning)
**File**: [nn_torch_ver/trainer.py](nn_torch_ver/trainer.py) → `decoding_train()`

- **Input**: Frozen learned representation `m` from Stage 1
- **Model**: `decoding` (energy-based score decoder)
- **Process**:
  1. Energy computation: `e = decoding(m)` → scalar energy per sample
  2. Score extraction: `score_hat = ∇_m e(m)` via autograd (requires `m.requires_grad=True`)
  3. Score field is gradient of energy w.r.t. representation
- **Loss**:
  - `score_loss`: Point-wise projection aligning `score_hat` with reference `score_ref`
  - `kl_loss`: KL-like divergence on pairwise distances in score space
- **Hyperparameters**: `beta` (KL weight), `tau` (temperature scaling)
- **Encoder frozen**: Only decoder parameters optimized

### Three-Kernel Robust Similarity
**File**: [nn_torch_ver/fitness.py](nn_torch_ver/fitness.py) → `compute_kernels(xo)`

Combines three kernels for robust representation matching:

1. **Log-RBF**: `1 / (1 + D²/median_D)` — geometry-aware, scale-normalized
2. **Cosine**: `Xn @ Xn.T` (unit norm) — direction-based similarity
3. **Covariance**: `X_centered @ X_centered.T / D` — variance-aware

**All kernels normalized via `safe_trace_normalize()`** (trace normalization + NaN handling). Prevents numerical instability in MIM loss.

---

## Tensor Naming Convention (Non-Negotiable)
Strict naming prevents shape mismatches and enables rapid code review. Always use:

- `x` — Numeric features only (N×D)
- `xo` — Rich view with one-hot type encoding (N×D+T)
- `t` — Wine type labels, integer (N,)
- `q` — Quality scores, float (N,)
- `m` — Learned representation/embedding (N×K)
- `g` — Adjacency matrix(ces): (N×N) single-channel OR (C×N×N) multi-channel
- `k` — Kernel similarity matrix (C×N×N) or (N×N)
- `score_hat`/`score_ref` — Score field gradients (N×D)

**Example error**: If you see shape mismatch in decoding, check that `m` is (N×K), not (N,).

---

## Module Organization

### PyTorch Version (Primary)
**File**: [nn_torch_ver/](nn_torch_ver/)

| Module | Role |
|--------|------|
| [data_process.py](nn_torch_ver/data_process.py) | `WineNNDataManager` (standardization), `WineNNDataset` (PyTorch wrapper), `build_type_graph()` |
| [nn_model.py](nn_torch_ver/nn_model.py) | `navie_nn` (encoder), `decoding` (decoder), building blocks (`navie_GCN`, `navie_fearue_head`, `aggloremation_layer`) |
| [trainer.py](nn_torch_ver/trainer.py) | `exp_runner()` (Stage 1), `decoding_train()` (Stage 2), loss functions |
| [fitness.py](nn_torch_ver/fitness.py) | `compute_kernels()`, kernel normalization, statistical aggregation |
| [selection.py](nn_torch_ver/selection.py) | Interaction matrices, advanced Laplacian kernels (experimental) |
| [dist.py](nn_torch_ver/dist.py) | Gaussian Mixture Modeling for reference score generation |

### JAX Version (Parallel Implementation)
**File**: [nn_jax_ver/](nn_jax_ver/) — Identical logic, JAX/Equinox API instead of PyTorch.

Use PyTorch version for new features. JAX mirrors implementation once PyTorch is stable.

### Classical Baseline
**File**: [classical_ml_model.py](classical_ml_model.py) — `KRR_Encoder` (scikit-learn Multi-Output Ridge). Exists for comparison; not actively used in current workflows.

---

## Data Flow

### From Raw Data to Score Field
```
CSV (wine data)
  ↓ WineNNDataManager.prepare()
  ├─ Standardize numeric features → x
  └─ One-hot encode type → xo
  ↓
Batch: {x, xo, t, q}
  ↓ build_type_graph(t) + build_type_graph(q)
  → g (stacked adjacency, 2×N×N)
  ↓ Stage 1: navie_nn(x, g) → m
  ↓ compute_kernels(xo) → k
  ↓ MIM + Stat loss — optimize encoder
  ↓
frozen navie_nn + fresh decoding()
  ↓ decoding(m) → e(m) (energy)
  ↓ ∇_m e(m) → score_hat (via autograd)
  ↓ score_loss + kl_loss — optimize decoder
  ↓
Final score field {score_hat, m}
```

---

## Development Workflows

### Running Experiments

**Active Workflow** — [workflow_nn_torch.ipynb](workflow_nn_torch.ipynb):
1. Load red + white wine data via `WineNNDataManager`
2. Create DataLoader with batch size (typically 32 or 64)
3. Train encoder with `exp_runner()` (Stage 1)
4. Freeze encoder, train decoder with `decoding_train()` (Stage 2)
5. Extract score fields and validate

**JAX Alternative** — [workflow_nn_jax.ipynb](workflow_nn_jax.ipynb): Same pipeline, JAX backend.

**Baseline (Incomplete)** — [workflow_classical_ml.ipynb](workflow_classical_ml.ipynb): For future use.

### Standard Import Pattern
```python
from nn_torch_ver.data_process import WineNNDataManager, build_type_graph
from nn_torch_ver.trainer import exp_runner, decoding_train
from nn_torch_ver.fitness import compute_kernels
from nn_torch_ver.nn_model import navie_nn, decoding
```

### Device Management
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# In loop: batch["x"] = batch["x"].to(device)
# Model: model.to(device)
```

---

## Type Annotations with JAXTyping
All forward signatures document tensor shapes for rapid review:

```python
def forward(self, x: Float[Array, 'n m'], g: Float[Array, 'c n n']) -> Float[Array, 'n k']:
    # n=batch, m=features, c=channels, k=representation_dim
```

**Convention**: Always specify batch (`n`), feature (`m`), and channel (`c`) dimensions. Prevents shape errors and aids code review.

---

## Critical Pitfalls

1. **Graph shape mismatch**: `g` must be (N×N) OR (C×N×N). Verify `build_type_graph()` output shape before passing to encoder.
2. **Gradient requirement in decoding**: `m` must have `requires_grad=True` to compute score gradients via `∇_m e(m)`.
3. **Kernel normalization**: Never skip `safe_trace_normalize()` in loss computation—prevents NaN propagation.
4. **Device consistency**: Batch tensors and model parameters must be on same device. Common error: model on GPU but batch on CPU.
5. **Tensor naming**: Use strict conventions (`x`, `xo`, `m`, `g`, `k`) to avoid confusion during code review. Typos `navie_` and `fearue_` are intentional—preserve for experiment reproducibility.

---

## External Dependencies
- **PyTorch** (`torch`, `nn`, `nn.functional`): Core NN framework, autograd
- **JAX/Equinox**: Alternative NN framework, functional programming style
- **scikit-learn** (`StandardScaler`, `OneHotEncoder`, `KernelRidge`): Preprocessing, kernels
- **einops** (`einsum`, `rearrange`, `repeat`): Tensor operations with compact notation
- **jaxtyping** (`Float[Array, '...']`): Type annotations with dimension tracking
- **pandas/numpy**: Data loading and numerical computing

---

## Quick Reference: Key Functions

| Function | Purpose | File |
|----------|---------|------|
| `WineNNDataManager` | Standardize + encode wine data | [data_process.py](nn_torch_ver/data_process.py) |
| `build_type_graph(t)` | Create binary adjacency matrix from type labels | [data_process.py](nn_torch_ver/data_process.py) |
| `navie_nn` | Graph-aware encoder | [nn_model.py](nn_torch_ver/nn_model.py) |
| `decoding` | Energy-based score decoder | [nn_model.py](nn_torch_ver/nn_model.py) |
| `exp_runner()` | Train encoder (Stage 1) | [trainer.py](nn_torch_ver/trainer.py) |
| `decoding_train()` | Train decoder (Stage 2) | [trainer.py](nn_torch_ver/trainer.py) |
| `compute_kernels(xo)` | 3-kernel similarity matrix | [fitness.py](nn_torch_ver/fitness.py) |
| `safe_trace_normalize()` | Normalize kernel + handle NaN | [fitness.py](nn_torch_ver/fitness.py) |
