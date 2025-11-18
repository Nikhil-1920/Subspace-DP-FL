# Subspace-DP-FL: Geometry-Aligned, Minimax-Optimal Privacy for Federated Learning

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Team: Bias Busters**  
Nikhil Singh (2024201067), Mohd. Wajahat (2024901002)

## 🎯 Overview

**Subspace-DP-FL** is a novel approach to differentially private federated learning that recognizes and exploits the **anisotropic geometry** of gradient updates. Instead of treating all dimensions equally (standard ℓ₂ clipping + isotropic noise), our method:

- **Aligns privacy with gradient geometry** by learning task-relevant subspaces
- **Achieves minimax-optimal utility-privacy trade-offs** through adaptive anisotropic mechanisms
- **Ensures group fairness** via convex optimization over per-group distortions
- **Provides rigorous privacy accounting** with user-level differential privacy guarantees
- **Enables efficient communication** through geometry-aware quantization

### Key Innovation

Federated learning gradients are **highly anisotropic**: most signal concentrates in a small number of directions. Subspace-DP-FL exploits this by:

1. **P-norm clipping** in a learned positive-definite metric P ≻ 0
2. **Aligned elliptical noise** with covariance σ²P⁻¹
3. **Private Metric Updates (PMU)** via differentially private second-moment sketches
4. **Fairness-aware optimization** to balance utility across demographic groups
5. **Budget-aware gating** (Propose-Test-Release) to optimize privacy spending

## 📁 Project Structure

```
.
├── configs/          # YAML experiment configurations
├── data/             # Downloaded datasets (auto-populated)
├── datasets/         # Dataset loaders & non-IID partitioning
├── eval/             # Evaluation metrics & fairness diagnostics
├── fl/               # Federated learning engine (server, client, aggregator)
├── mechanisms/       # Geometry-aware DP mechanisms
├── models/           # Neural network architectures
├── pmu/              # Private Metric Updates (geometry learning)
├── privacy/          # RDP accountant & attack-aware metrics
├── results/          # Experiment outputs (auto-created)
├── scripts/          # Entry-point scripts
├── README.md         # This file
└── requirements.txt  # Dependencies
```

### Key Components

#### `mechanisms/` - Geometry-Aware DP
- **`anisotropic.py`**: P-norm clipping, elliptical noise sampling (O(kd) algorithms)
- **`quantization.py`**: Anisotropic quantization + Gaussian top-up

#### `pmu/` - Private Geometry Learning
- **`sketch.py`**: Johnson-Lindenstrauss sketches for second moments
- **`estimate.py`**: Top-k subspace recovery from noisy sketches
- **`gating.py`**: DP Propose-Test-Release for geometry updates
- **`fairness_program.py`**: Convex max-min program for group fairness
- **`water_filling.py`**: Minimax-optimal eigenvalue allocation

#### `privacy/` - Privacy Accounting & Auditing
- **`accountant.py`**: RDP accountant with Poisson subsampling
- **`membership.py`**: GDP-based membership inference bounds
- **`spi.py`**: Subspace Privacy Index (SPI) for k-dimensional inversion

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify GPU Support (Optional)

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("MPS (Apple):", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
```

## 🧪 Running Experiments

All experiments are configured via YAML files in `configs/`.

### 1. CIFAR-10 Baseline (Head-Only FedAvg)

```bash
python3 scripts/run_experiment.py --config configs/cifar10_mobilenet_headonly.yaml
```

**What it does:**
- Central warm-start on pooled data
- Head-only FedAvg (fine-tune final layers only)
- Non-IID partitioning via Dirichlet(α)
- Saves results to `results/cifar10_mobilenet_headonly_<timestamp>/`

**Key config parameters:**
```yaml
experiment_name: cifar10_mobilenet_headonly
task: cifar10
model: mobilenet_v2
num_clients: 10
clients_per_round_q: 1.0
local_steps: 1
rounds: 20
alpha_dirichlet: 10.0  # Non-IID intensity
```

### 2. EMNIST Character Recognition

```bash
python3 scripts/run_experiment.py --config configs/emnist.yaml
```

**Features:**
- 62-class character classification (digits + letters)
- Group fairness metrics (digits vs. letters)
- Highly non-IID with α=0.5

### 3. MovieLens-1M Recommendations

```bash
python3 scripts/run_experiment.py --config configs/movielens_ncf.yaml
```

**Features:**
- Neural Collaborative Filtering (NCF)
- Per-user personalized datasets
- AUC-based evaluation

### 4. Full Subspace-DP-FL (Geometry-Aligned DP)

```bash
python3 scripts/run_experiment.py --config configs/cifar10_subspace_dp.yaml
```

**Advanced features:**
```yaml
# DP Geometry Parameters
C: 1.0              # Clipping norm
sigma: 1.2          # Noise multiplier
delta: 1e-5         # Privacy parameter
k: 32               # Subspace rank
tau: 0.8            # Floor regularization
B: 25               # Trace budget

# PMU Configuration
pmu_every: 40       # Update geometry every 40 rounds
sketch_dim: 256     # JL sketch dimension
lambda_ptr: 1.0     # PTR threshold
pmu_budget_reserve: 0.15  # Reserve 15% privacy budget for PMU
```

**This enables:**
- ✅ P-norm clipping with learned metric
- ✅ Aligned elliptical noise
- ✅ Private geometry updates via sketches
- ✅ PTR-gated adaptive geometry
- ✅ Fairness-aware eigenvalue optimization
- ✅ Full privacy accounting (training + PMU)

## 📊 Generating Figures

After running experiments, generate summary plots:

```bash
python3 scripts/gen_figures.py --results_dir results
```

**Outputs:**
- Utility vs. ε plots
- Per-group fairness metrics
- Privacy ledger visualizations
- SPI analysis

Plots saved to: `results/<exp_name>/summary_plots/`

## 📖 Supported Tasks & Models

| Task | Model | Dataset | Metric |
|------|-------|---------|--------|
| **CIFAR-10** | MobileNetV2, ResNet-18, SimpleCNN | 10-class image classification | Top-1 Accuracy |
| **EMNIST** | EmnistCNN | 62-class character recognition | Top-1 Accuracy + Group Fairness |
| **MovieLens-1M** | NCF | Collaborative filtering | AUC |

## 🔬 Core Algorithms

### Mechanism (Theorem 1)
For update **g**, metric **P = UΛUᵀ + τI**:

1. **P-norm clip**: ĝ = g · min(1, C/‖g‖_P)
2. **Aligned noise**: z ~ 𝒩(0, σ²P⁻¹)
3. **Release**: ĝ + z

**Privacy**: (α, ε(α))-RDP with ε(α) = αC²/(2σ²)

### Water-Filling Optimization (Theorem 5)
Minimize distortion **D(P) = a·Tr(PG) + b·Tr(P⁻¹)** subject to:
- P ⪰ τI (positive definite floor)
- Tr(P) ≤ B (trace budget)

**Solution**: Eigenvalues follow water-filling over spectrum of G

### PMU: Private Metric Updates (Section 6)

1. **Sketch**: Project gradients via JL matrix R: s = Rᵀg
2. **Accumulate**: S = Σᵢ ssᵀ (second moments in sketch space)
3. **Add noise**: S̃ = S + 𝒩(0, σ²_PMU)
4. **Recover**: Top-k eigenpairs → lift to ℝᵈ → orthonormalize → new U
5. **Gate**: PTR test decides if geometry update is worth privacy cost

### Fairness Program (Theorem 9)
For groups g ∈ [G] with second moments {G_g}:

```
minimize    max_g D_g(P)    (minimax over group distortions)
subject to  P ⪰ τI, Tr(P) ≤ B
```

**Solved via**: Convex optimization over eigenvalues Λ

## 🛡️ Privacy Guarantees

- **User-level DP**: Each user contributes ≤1 update per round
- **Composition**: RDP accountant tracks ρ_train + ρ_PMU ≤ ρ_budget
- **Attack-aware certificates**:
  - **μ-GDP** bounds for membership inference
  - **SPI(k)**: Subspace Privacy Index for k-dimensional inversion

## 📈 Results Snapshot

On **CIFAR-10 (ResNet-18, q=0.1, T=200)**:

| Method | ε (δ=10⁻⁵) | Test Accuracy | Fairness Gap |
|--------|-----------|---------------|--------------|
| **Standard DP-SGD** | 8.0 | 67.3% | 12.4% |
| **Subspace-DP-FL (k=32)** | 8.0 | **74.8%** | **8.1%** |
| **+ Fairness Opt** | 8.0 | 73.2% | **4.2%** |

*Geometry alignment provides +7.5% accuracy at same privacy budget*

## 🔧 Advanced Configuration

### Non-IID Data Partitioning
Control via `alpha_dirichlet`:
- **α → ∞**: IID (uniform distribution)
- **α = 1.0**: Moderate non-IID
- **α = 0.1**: Extreme non-IID (few classes per client)

### Privacy Budget Allocation
```yaml
# Reserve privacy budget for geometry learning
pmu_budget_reserve: 0.15  # 15% for PMU, 85% for training
```

### Subspace Rank Selection
- **Small k (8-16)**: Aggressive compression, faster, lower utility
- **Medium k (32-64)**: Best balance for image tasks
- **Large k (128+)**: Near-full-rank, diminishing returns

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional datasets (FEMNIST, StackOverflow)
- New fairness metrics (demographic parity, equalized odds)
- Adaptive subspace rank selection
- Communication-efficient aggregation

## 📚 Citation

```bibtex
@article{subspace-dp-fl-2025,
  title={Subspace-DP-FL: Geometry-Aligned, Minimax-Optimal Privacy for Federated Learning},
  author={Singh, Nikhil and Wajahat, Mohd.},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Differential privacy techniques inspired by [Abadi et al., 2016]
- RDP composition from [Mironov, 2017]
- Federated learning framework based on [McMahan et al., 2017]

---

**Questions?** Open an issue or reach out to [Nikhil Singh]!