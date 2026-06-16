# DPDNet-S + CauSiam

Official PyTorch implementation of **Continual Test-Time Adaptation for Single Image Defocus Deblurring via Causal Siamese Networks** (*International Journal of Computer Vision*, **IJCV 2025**).

This repository implements **CauSiam** (Causal Siamese Networks) as a plug-and-play continual test-time adaptation (CTTA) framework on top of the source defocus deblurring model **DPDNet-S**. During inference, CauSiam adapts the source model online to continuously changing target domains using only unlabeled test images, while CLIP (ViT-B/32) provides universal semantic priors to improve causal identifiability between blurry inputs and restored images.

---

## Paper Information

**Title:** Continual Test-Time Adaptation for Single Image Defocus Deblurring via Causal Siamese Networks
**Venue:** International Journal of Computer Vision (**IJCV**), 2025

**ArXiv:** [2501.09052](https://arxiv.org/abs/2501.09052)

### Abstract (Brief)

Single image defocus deblurring (SIDD) restores all-in-focus images from defocused ones. Performance drops under out-of-distribution testing mainly due to **lens-specific point spread function (PSF) heterogeneity**. Existing CTTA methods based on entropy minimization are poorly suited to pixel-level regression. CauSiam addresses this with:

1. **Siamese consistency learning** for online CTTA on SIDD
2. **CLIP-guided semantic priors** injected via cross-attention (VSPI module)
3. **EMA teacher + stochastic restoration** for stable long-term adaptation

---

## Method Overview

### Problem Setting

| Setting | Source data at test time | Target data | Distribution |
|---------|--------------------------|-------------|--------------|
| CTTA (CauSiam) | Pretrained model only | Unlabeled test stream | Continuously changing |

The test stream follows the CTTA protocol in the paper: **DPDD → RealDOF → LFDOF → RTF** (and optionally extended variants). Each dataset corresponds to a different camera / PSF distribution.

### DPDNet-S (Source Model)

**DPDNet-S** is the compact U-Net-style encoder-decoder from [Abuolaim & Brown, 2020](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel). The [official release](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) is implemented in **TensorFlow**; we **reimplemented DPDNet-S in PyTorch** for this project and trained it on the DPDD training set. In this repo it serves as the **frozen-pretrained source backbone** loaded via `--resume_model`, then wrapped by CauSiam at test time.

### CauSiam Pipeline (This Codebase)

For each test image, `Clip_TTA` in `da_clip.py` performs:

```
Input blur image x
    │
    ├─► Source copy (model_int) ──► initial restoration ──► CLIP ViT-B/32 ──► semantic context s
    │
    ├─► EMA teacher (model_ema) ──► N=4 geometric TTA views ──► consistency target (mean)
    │
    └─► Online student (model) ──► DPDNet-S + Cross-Attention at conv6 (conditioned on s)
              │
              ├─ Loss: L1(student output, EMA teacher mean)
              ├─ Update: attention weights & bias only (Adam, lr=1e-4)
              ├─ EMA teacher update (mt=0.9)
              └─ Stochastic restore (rst=0.01)
```

**Key modules in code:**

| Paper module | Code location | Description |
|--------------|---------------|-------------|
| Source SIDD model | `model/archs/network.py` | DPDNet-S U-Net |
| VSPI (semantic prior integration) | `network.py` → `CrossAttention` at conv6 | CLIP features fused with `0.05 * attention + conv6` |
| Siamese consistency | `da_clip.py` → `forward_adapt_network` | L1 loss between student and EMA-augmented mean |
| EMA teacher | `da_clip.py` → `update_ema_variables` | Exponential moving average, `mt=0.9` |
| Stochastic restore | `da_clip.py` | Random reset of adapted params (CoTTA-style) |
| CLIP encoder | `da_clip.py` → `clip.load("ViT-B/32")` | Frozen ViT-B/32 image encoder |

---

## Project Structure

```
dpdnet_clip_vit32_bais-iteration1ci/
├── train.py                 # Train DPDNet-S from scratch (optional)
├── val.py                   # Validate checkpoints during training
├── test.py                  # Single-dataset test with CauSiam
├── test_causiam.py          # Multi-dataset CTTA evaluation (main benchmark script)
├── da_clip.py               # CauSiam core: Clip_TTA, augmentations, EMA, restore
├── model/
│   ├── model.py             # Model wrapper, TTA setup, train/test API
│   ├── tta_config.py        # ★ TTA hyper-parameters (edit here)
│   └── archs/
│       └── network.py       # DPDNet-S + cross-attention at conv6
├── dataset/
│   ├── mydataset.py         # DPDD / RealDOF / LFDOF / RTF loader
│   └── sid_dataset.py       # SID dataset (optional)
├── loss/                    # MSE training loss for source model
├── options/
│   ├── dpdd/option.py       # Paths, batch size, resume checkpoint
│   └── model_para.py        # Dropout rate, etc.
├── utils/                   # Unified utilities (merged from util/ + utils/)
│   ├── eval_utils.py        # Inference & evaluation pipeline
│   ├── run_utils.py         # Logging, seeding, CLI parsing
│   ├── img_utils.py         # Image I/O helpers
│   ├── metric_cal.py        # PSNR / SSIM computation
│   ├── metric_utils.py
│   └── image_io.py          # Low-level image read/crop
├── scripts/                 # Auxiliary tools (non-core)
│   ├── datatest.py          # Dataset sanity check
│   ├── count_model_params.py
│   ├── clip_score.py
│   ├── lr.py
│   └── resume_lr.py
├── CLIP/                    # CLIP dependency (ViT-B/32)
├── open_clip/               # open_clip (optional / legacy)
├── checkpoints/             # DPDNet-S weights (download dpdnet-s.pt here)
└── clip_model/              # CLIP weights (auto-downloaded on first run)
```

---

## Requirements

### Environment

- Python 3.8+
- PyTorch 1.10+ with CUDA

### Python Packages

```bash
pip install torch torchvision
pip install numpy opencv-python pillow tqdm einops thop
pip install ftfy regex tqdm   # for CLIP
```

Install CLIP dependencies as needed by the bundled `CLIP/` module.

### Pretrained Weights

| Asset | Purpose | How to obtain |
|-------|---------|---------------|
| **DPDNet-S** | Source backbone (PyTorch) | [Download dpdnet-s.pt](https://drive.google.com/file/d/1VUjoSZkk-p5R18JegZlPpeJ3J9xDH0EZ/view?usp=sharing) |
| **CLIP ViT-B/32** | Semantic prior encoder | Auto-downloaded to `./clip_model/` on first run |

#### DPDNet-S (`dpdnet-s.pt`)

The original DPDNet-S code from [Abuolaim & Brown, 2020](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) is written in **TensorFlow**. We reimplemented the same architecture in **PyTorch**, trained it on the DPDD training set, and release the checkpoint as `dpdnet-s.pt` (~119 MB).

**Download:** [Google Drive — dpdnet-s.pt](https://drive.google.com/file/d/1VUjoSZkk-p5R18JegZlPpeJ3J9xDH0EZ/view?usp=sharing)

**Setup:**

```bash
mkdir -p checkpoints
# Download dpdnet-s.pt into checkpoints/, then run:
python test_causiam.py --resume_model checkpoints/dpdnet-s.pt
```

Alternatively, update `DEFAULT_RESUME_MODEL` in `options/dpdd/option.py` to your local path.

You can also train your own checkpoint from scratch with `python train.py --save exp/dpdd` (saved to `exp/dpdd/train-YYYYMMDD-HHMMSS/model_state/best.pt`).

#### CLIP ViT-B/32

Loaded automatically by `da_clip.py` via `clip.load("ViT-B/32", download_root="./clip_model/")`. Weights (~338 MB) are cached under `./clip_model/ViT-B-32.pt` on first run. No manual download required.

---

## Dataset Preparation

CauSiam evaluation uses four public SIDD benchmarks in CTTA order:

| Dataset | Images (test) | Device / note | PSF shift type |
|---------|---------------|---------------|----------------|
| **DPDD** | 76 | Canon (same family as training) | Lens-agnostic |
| **RealDOF** | 50 | Sony α7R IV | Lens-specific |
| **LFDOF** | 725 | Lytro Illum | Lens-specific |
| **RTF** | 22 | Real scene | Lens-specific |

### Directory Layout

Organize data as follows (modify paths in `test_causiam.py` and `options/dpdd/option.py`):

```
Defocus_Deblur_Dataset_Test/
├── DPDD/test/
│   ├── input/
│   └── target/
├── RealDOF/test/
│   ├── input/
│   └── target/
├── LFDOF/test/
│   ├── input/
│   └── target/
└── RTF/test/
    ├── input/
    └── target/
```

Default root in code:

```
/home/csh/dataset/Defocus_Deblur_Dataset_Test/
```

### DPDD Training Set (for training DPDNet-S)

```
dd_dp_dataset_canon_patch/train_c/source/
dd_dp_dataset_canon_patch/train_c/target/
```

---

## Configuration

### TTA Hyper-parameters (`model/tta_config.py`)

These control CauSiam behavior. They correspond to the paper's implementation details (Section 4.2):

| Variable | Default | Paper symbol | Description |
|----------|---------|--------------|-------------|
| `TTA_STEPS` | `1` | K | Adaptation iterations per image |
| `TTA_LR` | `1e-4` | lr | Adam learning rate |
| `TTA_ADAM_BETA` | `0.9` | β₁ | Adam β₁ |
| `TTA_MT_ALPHA` | `0.9` | η | EMA decay for teacher |
| `TTA_RST_M` | `0.01` | — | Stochastic restoration probability |
| `TTA_AP` | `0.92` | — | Anchor parameter (reserved) |

**Also fixed in code (see `network.py` / `da_clip.py`):**

| Setting | Value | Paper |
|---------|-------|-------|
| Cross-attention scale α | `0.05` | α = 0.05 |
| CLIP backbone | ViT-B/32 | ViT-B/32 |
| Trainable params | `attention` weights & bias | CA module only |
| TTA augmentations | 4 rotations | N = 5 in paper; this release uses 4 |
| LR scheduler | StepLR, γ=0.95 | — |

### Test Script Settings (`test_causiam.py`)

```python
DATASET_NAMES = ['DPDD', 'RealDOF', 'LFDOF', 'RTF']
DATASET_ROOT = '/home/csh/dataset/Defocus_Deblur_Dataset_Test/'
SNAPSHOT_ROOT = '260616-result-CAUSIAM-DPDNet-Final'
DEFAULT_GPU = '5'
DEFAULT_SEED = 42
```

### Source Model Options (`options/dpdd/option.py`)

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--resume_model` | `checkpoints/dpdnet-s.pt` | DPDNet-S checkpoint (see [Pretrained Weights](#pretrained-weights)) |
| `--val_batch_size` | 1 | Batch size (must be 1 for CTTA) |
| `--dropout_rate` | 0.4 | Dropout in DPDNet-S encoder |
| `--y_channel` | 0 | 0 = RGB PSNR, 1 = Y-channel PSNR |
| `--seed` | 0 (42 in test_causiam) | Random seed |

---

## Usage

All commands should be run from the project root:

```bash
cd dpdnet_clip_vit32_bais-iteration1ci
```

### 1. CTTA Benchmark (Main — DPDNet-S + CauSiam)

Evaluate on all four datasets sequentially (continual adaptation):

```bash
python test_causiam.py --gpu 0
```

**Outputs:**

```
260616-result-CAUSIAM-DPDNet-Final/
├── test.log              # PSNR log for all datasets
├── DPDD/                 # Deblurred PNGs
├── RealDOF/
├── LFDOF/
└── RTF/
```

Modify `SNAPSHOT_ROOT` and `DATASET_ROOT` in `test_causiam.py` before running.


## Expected Results

### DPDNet-S + CauSiam (ViT-B/32, Table 7 in paper)

| Dataset | PSNR (dB) | SSIM |
|---------|-----------|------|
| DPDD | 24.862 | 0.759 |
| RealDOF | 23.713 | 0.687 |
| LFDOF | 25.617 | 0.778 |
| RTF | 24.399 | 0.785 |
| **Average** | **25.412** | **0.771** |

### Source-only DPDNet-S (no adaptation, baseline)

| Dataset | PSNR (dB) |
|---------|-----------|
| DPDD | 24.648 |
| RealDOF | 23.254 |
| LFDOF | 24.810 |
| RTF | 23.578 |
| Average | 24.676 |


## Implementation Notes

1. **Continual adaptation:** Unlike episodic TTA, the model state carries across datasets. DPDD is processed first, then RealDOF, LFDOF, RTF without resetting weights — matching the CTTA protocol in the paper.

2. **Only attention is trainable:** `da_clip.configure_model()` enables gradients only for modules whose names contain `attention`. This corresponds to updating the CA (Cross-Attention) module only.

3. **Large images:** For images exceeding 1680×1150, a center crop of 1120×1120 is applied during adaptation; full-resolution inference uses the original size for the final EMA ensemble (see `flag` branch in `da_clip.py`).

4. **CLIP loading:** Weights are downloaded to `./clip_model/` via the bundled CLIP library on first execution. Ensure network access or place weights manually.

5. **Metric computation:** PSNR is computed on RGB channels by default (`y_channel=0`), consistent with the paper's full-reference evaluation on DPDD / RealDOF / LFDOF / RTF.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{cui2025causiam,
  title={Continual Test-Time Adaptation for Single Image Defocus Deblurring via Causal Siamese Networks},
  author={Cui, Shuang and Li, Yi and Li, Jiangmeng and Tang, Xiongxin and Su, Bing and Xu, Fanjiang and Xiong, Hui},
  journal={International Journal of Computer Vision},
  year={2025},
  note={arXiv:2501.09052}
}
```

### Related Work

```bibtex
@inproceedings{abuolaim2020defocus,
  title={Defocus deblurring using dual-pixel data},
  author={Abuolaim, Abdullah and Brown, Michael S},
  booktitle={ECCV},
  year={2020}
}

@inproceedings{radford2021clip,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}

@inproceedings{wang2022cotta,
  title={Continual test-time domain adaptation},
  author={Wang, Qin and Fink, Olga and Van Gool, Luc and others},
  booktitle={CVPR},
  year={2022}
}
```

---

## License

This project builds upon DPDNet-S, CLIP, and CoTTA-style adaptation code. Please refer to the respective licenses of the upstream repositories when distributing or modifying this code.

---
