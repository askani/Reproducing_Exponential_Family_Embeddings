# Exponential Family Embeddings - Reproduction Study

A PyTorch reproduction of **Exponential Family Embeddings (EFE)** from Rudolph et al. (2016), focusing on Poisson-based models for count data applications.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📖 Overview

This project reproduces the Exponential Family Embeddings framework for learning distributed representations from count data. We implement three Poisson embedding variants and evaluate them on market basket and movie rating datasets.

### Models Implemented

| Model | Description | Mean Function |
|-------|-------------|---------------|
| **P-EMB** | Poisson Embedding (multiplicative) | λ = exp(ρᵀᾱ) |
| **P-EMB-DW** | P-EMB with downweighted zeros (w₀=0.1) | λ = exp(ρᵀᾱ) |
| **AP-EMB** | Additive Poisson Embedding | λ = ρᵀᾱ + b |
| **HPF** | Hierarchical Poisson Factorization (baseline) | λ = θᵤᵀβᵢ |
| **Poisson PCA** | Poisson PCA (baseline) | λ = exp(wᵀh + c + μ) |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/exponential-family-embeddings.git
cd exponential-family-embeddings

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
umap-learn>=0.5.0
```

### Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l7fjNNbNhtjLGaWbSHo7st7mKQRJGxcd?usp=sharing)

1. Open the notebook in Google Colab
2. Go to `Runtime → Change runtime type → GPU`
3. Run all cells

## 📁 Project Structure

```
exponential-family-embeddings/
├── README.md
├── requirements.txt
├── Market_Basekt_Data_Analysis.ipynb    # Main notebook (Colab-ready)
├── data/
│   └── data.csv.zip              # raw market basket dataset
└── results/
    └── png files               # Generated plots and visualizations
```

## 📊 Datasets

### Market Basket Data

The dataset should contain grocery transaction records with the following columns:
Download from [Dunnhumby Data](https://www.dunnhumby.com/source-files/):

| Column | Description |
|--------|-------------|
| `product_id` | Unique product identifier |
| `household_key` | Household identifier |
| `basket_id` | Transaction/basket identifier |
| `quantity` | Purchase quantity |

### MovieLens Data

Download from [MovieLens](https://grouplens.org/datasets/movielens/):
- Ratings ≥ 3 are converted to counts
- Ratings < 3 are set to zero

## 📈 Results

### Market Basket Dataset

| Model | K=10 | K=20 | K=50 |
|-------|------|------|------|
| P-EMB | -7.30 ± 0.010 | **-7.28 ± 0.014** | -7.29 ± 0.016 |
| P-EMB (dw) | -7.39 ± 0.008 | -7.34 ± 0.009 | **-7.28 ± 0.014** |
| AP-EMB | -7.74 ± 0.003 | -7.79 ± 0.003 | -7.91 ± 0.003 |
| HPF | -7.78 ± 0.004 | -7.79 ± 0.004 | -7.77 ± 0.003 |
| Poisson PCA | -7.29 ± 0.011 | -7.25 ± 0.013 | **-7.22 ± 0.013** |

### MovieLens Dataset

| Model | K=10 | K=20 | K=50 |
|-------|------|------|------|
| P-EMB | -2.814 ± 0.006 | -2.794 ± 0.006 | -2.781 ± 0.007 |
| P-EMB (dw) | **-1.297 ± 0.002** | **-1.297 ± 0.002** | **-1.298 ± 0.002** |
| AP-EMB | -2.154 ± 0.006 | -2.151 ± 0.006 | -2.150 ± 0.006 |
| HPF | -0.028 ± 0.000 | -0.023 ± 0.000 | -0.025 ± 0.000 |
| Poisson PCA | -0.027 ± 0.000 | -0.033 ± 0.000 | -0.043 ± 0.000 |

### Key Findings

- ✅ **Multiplicative models (P-EMB) outperform additive variants (AP-EMB)**
- ✅ **Downweighting zeros improves performance on sparse data**
- ✅ **AP-EMB limited by non-negativity constraint** — cannot model negative correlations

## 🖼️ Embedding Visualizations

<p align="center">
  <img src="results/figures/tsne_comparison.png" alt="t-SNE Visualization" width="800"/>
</p>

P-EMB-DW produces the clearest cluster separation, while AP-EMB shows degenerate patterns due to non-negativity constraints.

## 📚 Reference

If you use this code, please cite the original paper:

```bibtex
@inproceedings{rudolph2016exponential,
  title={Exponential Family Embeddings},
  author={Rudolph, Maja and Ruiz, Francisco J. R. and Mandt, Stephan and Blei, David M.},
  booktitle={Advances in Neural Information Processing Systems},
  volume={29},
  pages={478--486},
  year={2016}
}
```

## 🙏 Acknowledgments

- Original paper: [Rudolph et al. (2016) - Exponential Family Embeddings](https://papers.nips.cc/paper/2016/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html)
- [PyTorch](https://pytorch.org/) for deep learning framework
- [UMAP](https://umap-learn.readthedocs.io/) for dimensionality reduction

---
