# Phase B: Architecture Baseline Comparison

> **Date:** 2026-06-16  
> **Backend:** LibTorch GPU (RTX 4060 Laptop, 8 GB VRAM, CUDA 13.0)  
> **Dataset:** GTSRB 43-class (39,209 training, 12,630 test)  
> **Train/Val Split:** Track-aware, class-stratified 80/20 (30,869 / 8,340)  
> **Branch:** `claude_release`

---

## 1. Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 0.0001 |
| Seed | 42 |
| Class balancing | Yes |
| Validation split | 20%, track-based |
| LR scheduler | None (constant LR) |

### Per-Experiment Settings

| ID | Architecture | Params | Augmentation | Model File |
|----|-------------|--------|-------------|------------|
| B1 | LeNet (2 Conv + 2 FC) | 56,195 | No | `gtsrb_v1_lenet_plain.pt` |
| B2 | LeNet (2 Conv + 2 FC) | 56,195 | Yes | `gtsrb_v2_lenet_aug.pt` |
| B3 | Enhanced (3 Conv + 3 FC) | 1,286,251 | No | `gtsrb_v3_enhanced_plain.pt` |
| B4 | Enhanced (3 Conv + 3 FC) | 1,286,251 | Yes | `gtsrb_v4_enhanced_aug.pt` |

---

## 2. Results Summary

### Validation Set (Best Epoch)

| ID | Best Val Acc | Best Epoch | Mean Class Acc | Min Class Acc | Final Val Acc |
|----|-------------|-----------|---------------|--------------|--------------|
| B1 | **89.69%** | 4 | 85.03% | 16.67% | 89.46% |
| B2 | **89.90%** | 5 | 86.57% | 25.00% | 89.90% |
| B3 | **94.53%** | 5 | 90.98% | 45.00% | 94.53% |
| B4 | **95.12%** | 5 | 91.81% | 50.00% | 95.12% |

### Official Test Set (Best Checkpoint, for reference)

| ID | Test Acc | Mean Class Acc | Min Class Acc | Test Loss |
|----|---------|---------------|--------------|-----------|
| B1 | 87.99% | 83.96% | 30.00% | 0.989 |
| B2 | 87.01% | 85.72% | 42.22% | 0.565 |
| B3 | **93.34%** | 91.63% | 46.67% | 0.307 |
| B4 | 92.69% | 91.01% | 45.00% | 0.297 |

---

## 3. Per-Epoch Trajectory

### B1: LeNet, No Augmentation

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Mean Class | Min Class | Duration |
|-------|-----------|----------|---------|---------|-----------|----------|---------|
| 1 | 1.2451 | 67.11% | 0.8873 | 80.34% | 77.66% | 1.67% | 310.7s |
| 2 | 0.1646 | 95.52% | 0.7666 | 86.70% | 82.24% | 11.11% | 73.0s |
| 3 | 0.0814 | 97.89% | 0.6776 | 87.79% | 84.02% | 13.33% | 36.7s |
| 4 | 0.0518 | 98.66% | 0.7387 | **89.69%** | 85.03% | 16.67% | 27.1s |
| 5 | 0.0401 | 98.95% | 0.6622 | 89.46% | 85.72% | 20.00% | 26.0s |

### B2: LeNet, With Augmentation

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Mean Class | Min Class | Duration |
|-------|-----------|----------|---------|---------|-----------|----------|---------|
| 1 | 1.9901 | 45.33% | 0.9482 | 71.12% | 70.84% | 1.67% | 42.3s |
| 2 | 0.6417 | 80.65% | 0.6829 | 80.77% | 80.15% | 10.00% | 48.8s |
| 3 | 0.4329 | 86.87% | 0.5244 | 88.42% | 84.13% | 25.00% | 48.8s |
| 4 | 0.3431 | 89.70% | 0.5172 | 89.72% | 84.76% | 0.00% | 43.0s |
| 5 | 0.2862 | 91.29% | 0.4671 | **89.90%** | 86.57% | 25.00% | 51.8s |

### B3: Enhanced, No Augmentation

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Mean Class | Min Class | Duration |
|-------|-----------|----------|---------|---------|-----------|----------|---------|
| 1 | 3.1531 | 12.39% | 1.7931 | 39.35% | 39.16% | 0.00% | 21.6s |
| 2 | 0.7454 | 75.51% | 0.5062 | 85.88% | 82.76% | 5.00% | 23.3s |
| 3 | 0.1508 | 95.21% | 0.3368 | **91.51%** | 87.11% | 36.67% | 28.0s |
| 4 | 0.0718 | 97.83% | 0.2641 | 93.55% | 90.03% | 42.22% | 26.5s |
| 5 | 0.0415 | 98.75% | 0.2617 | **94.53%** | 90.98% | 45.00% | 23.6s |

### B4: Enhanced, With Augmentation

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Mean Class | Min Class | Duration |
|-------|-----------|----------|---------|---------|-----------|----------|---------|
| 1 | 3.3649 | 8.14% | 2.3626 | 23.50% | 21.11% | 0.00% | 48.7s |
| 2 | 1.7794 | 42.24% | 0.8812 | 70.01% | 70.33% | 3.33% | 58.9s |
| 3 | 0.6597 | 78.73% | 0.4043 | 87.54% | 84.88% | 31.67% | 57.6s |
| 4 | 0.3710 | 88.30% | 0.2544 | 93.06% | 90.35% | 50.00% | 68.6s |
| 5 | 0.2690 | 91.63% | 0.2543 | **95.12%** | 91.81% | 50.00% | 52.9s |

---

## 4. Speed Analysis

### Per-Epoch Timing

| Experiment | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Avg (steady) |
|-----------|---------|---------|---------|---------|---------|-------------|
| B1 (LeNet, -aug) | 310.7s | 73.0s | 36.7s | 27.1s | 26.0s | 26-37s |
| B2 (LeNet, +aug) | 42.3s | 48.8s | 48.8s | 43.0s | 51.8s | 43-52s |
| B3 (Enhanced, -aug) | 21.6s | 23.3s | 28.0s | 26.5s | 23.6s | 23-28s |
| B4 (Enhanced, +aug) | 48.7s | 58.9s | 57.6s | 68.6s | 52.9s | 50-69s |

### Total Wall Time

| Experiment | Total Time |
|-----------|-----------|
| B1 | 473.5s (7.9 min) |
| B2 | 234.7s (3.9 min) |
| B3 | 123.1s (2.1 min) |
| B4 | 286.9s (4.8 min) |
| **All 4** | **1,118.2s (18.6 min)** |

---

## 5. Architecture Comparison: LeNet vs Enhanced

### Accuracy

| Metric | LeNet (B1 best) | Enhanced (B3 best) | Improvement |
|--------|----------------|-------------------|-------------|
| Best Val Acc | 89.69% | **94.53%** | **+4.84%** |
| Mean Class Val Acc | 85.03% | **90.98%** | **+5.95%** |
| Min Class Val Acc | 16.67% | **45.00%** | **+28.33%** |
| Test Acc | 87.99% | **93.34%** | **+5.35%** |

Enhanced improves the worst-class accuracy from **16.67% → 45.00%** — critical for GTSRB's 10.7× class imbalance.

---

## 6. Data Augmentation Impact

| Architecture | No Aug (Best Val) | With Aug (Best Val) | Delta |
|-------------|------------------|-------------------|-------|
| LeNet | 89.69% | 89.90% | +0.21% |
| Enhanced | 94.53% | 95.12% | +0.59% |

- 5 epochs is insufficient for augmentation to fully manifest.
- Augmented models show much smaller train/val gaps (1.4% vs 9.5%), confirming reduced overfitting.
- Min class accuracy for Enhanced improves from 45% → 50% with augmentation.

---

## 7. Recommendations for Phase C

**Use Enhanced + augmentation for Phase C formal training.**

| Parameter | Recommended | Rationale |
|-----------|------------|-----------|
| Architecture | Enhanced | +4.84% over LeNet, faster per epoch on GPU |
| Epochs | 60-80 | Not plateauing at epoch 5 |
| Batch size | 64 | Proven |
| LR | 0.01 | Effective |
| Momentum | 0.9 | Standard |
| Weight decay | 0.0005 | Increase for formal training |
| Augmentation | Yes | Reduced overfitting, better min class |
| LR scheduler | StepLR(20, 0.1) | Fine-tuning in later epochs |
| Early stopping | Patience 10 | Prevent overfitting |

**Expected targets:** Val ≥ 96%, Test ≥ 96%, Min class ≥ 80%.

---

## 8. Model Artifacts

| File | Size | SHA-256 |
|------|------|---------|
| `gtsrb_v1_lenet_plain.pt` | 227 KB | `EFEFC9C2...` |
| `gtsrb_v1_lenet_plain_best.pt` | 227 KB | `47951CDB...` |
| `gtsrb_v2_lenet_aug.pt` | 226 KB | `137EED82...` |
| `gtsrb_v2_lenet_aug_best.pt` | 226 KB | `D168FA8C...` |
| `gtsrb_v3_enhanced_plain.pt` | 5,036 KB | `04CFFABF...` |
| `gtsrb_v3_enhanced_plain_best.pt` | 5,036 KB | `D2497F1B...` |
| `gtsrb_v4_enhanced_aug.pt` | 5,036 KB | `E850D626...` |
| `gtsrb_v4_enhanced_aug_best.pt` | 5,036 KB | `4B6A45A2...` |

Logs: `codex/logs/phase_b/B[1-4]_*.csv`
