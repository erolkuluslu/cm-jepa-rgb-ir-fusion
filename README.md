# Cross-Modal JEPA Gate-Blend Fusion for Lightweight RGB/Infrared Perception

This repository contains the final IEEE-style research package for a study of visible RGB and thermal infrared image fusion for lightweight object detection and semantic segmentation. The project evaluates whether a cross-modal attention fusion model with a JEPA-style representation objective can improve downstream perception under challenging visual conditions such as night, fog, rain, low light, and reduced contrast.

The main conclusion is deliberately nuanced: RGB/IR fusion is beneficial, but the simple average-fusion baseline is the strongest overall method in the final experiments. The proposed CM-JEPA gate-blend representation is competitive for object detection, especially under the stricter mAP@0.5:0.95 metric, but it does not preserve dense semantic information as well as average fusion for segmentation.

## Research Question

Can a cross-modal attention fusion model, strengthened with patch/token-level JEPA pretraining, improve lightweight object detection and semantic segmentation compared with RGB-only, IR-only, and classical average-fusion baselines?

## Contributions

- Built a complete RGB/IR fusion evaluation pipeline over 6025 aligned real image pairs from M3FD, MSRS, and RoadScene.
- Compared RGB-only, IR-only, average fusion, learned attention fusion variants, and the final CM-JEPA gate-blend representation.
- Evaluated fusion through downstream lightweight perception tasks instead of relying only on image-level fusion metrics.
- Used YOLO11n for detection on M3FD and U-Net with a MobileNetV2 encoder for semantic segmentation on MSRS.
- Repaired the final A7 evaluation protocol so that CM-JEPA metrics and downstream experiments use the same gate-guided RGB/IR blend rather than a collapsed raw decoder output.

## Method Overview

The proposed method is a mid-level RGB/IR fusion approach. RGB and IR inputs are first encoded with separate modality-specific branches. Channel attention recalibrates each modality, a spatial gate estimates where each modality should dominate, and a JEPA-style latent prediction objective encourages cross-modal representation consistency.

The final downstream representation is the gate-guided image-level blend:

```text
I_A7 = G_up * I_RGB + (1 - G_up) * rep(I_IR)
```

where `G_up` is the learned spatial gate upsampled to image resolution and `rep(I_IR)` is the infrared image replicated into three channels. This gate-blend representation is used consistently for YOLO detection, U-Net segmentation, and the repaired A7 fusion-quality metrics.

## Dataset Summary

The standardized manifest contains 6025 aligned RGB/IR pairs:

| Dataset | Role in this study | Pair count |
|---|---:|---:|
| M3FD | Object detection with YOLO11n | 4280 |
| MSRS | Semantic segmentation with U-Net/MobileNetV2 | 1524 |
| RoadScene | Fusion-quality and general fusion analysis | 221 |

The overall split contains 4215 training, 902 validation, and 908 test pairs. For the M3FD detection subset, the saved YOLO split contains 2994 training, 641 validation, and 645 test images for each evaluated representation.

Raw datasets are not included in this repository. The reproducibility notebook documents the expected dataset structure and experiment pipeline.

## Experimental Results

### Downstream Perception

| Method | mAP@0.5 | mAP@0.5:0.95 | mIoU |
|---|---:|---:|---:|
| RGB-only | 0.7616 | 0.4887 | 0.5307 |
| IR-only | 0.7064 | 0.4649 | 0.4976 |
| Average fusion | **0.7718** | **0.5115** | **0.5551** |
| CM-JEPA gate-blend | 0.7567 | 0.5080 | 0.4774 |

Average fusion gives the best overall downstream performance. CM-JEPA gate-blend is close to average fusion in mAP@0.5:0.95 and improves over RGB-only under that stricter detection metric, but it underperforms for semantic segmentation.

### Fusion-Quality and Ablation Metrics

| Method | Description | EN | SF | AG | NMI mean |
|---|---|---:|---:|---:|---:|
| A0 | RGB-only | 6.2097 | 0.04021 | 0.01517 | 1.5336 |
| A1 | IR-only | **7.6092** | 0.03879 | 0.01275 | 1.4436 |
| A2 | Average fusion | 6.8732 | 0.02836 | 0.01141 | 1.1617 |
| A3 | Concatenation, no gate | 7.3225 | 0.01873 | 0.01022 | 1.0744 |
| A4 | No channel attention | 7.3275 | 0.01898 | 0.01038 | 1.0715 |
| A5 | No spatial gate | 7.2397 | 0.01715 | 0.00916 | 1.0805 |
| A6 | Full attention, no JEPA | 7.2156 | 0.01826 | 0.00976 | 1.0754 |
| A7 | Full CM-JEPA gate-blend | 6.8898 | 0.02834 | 0.01138 | 1.1635 |

A3-A6 are fusion-quality ablations only. They were not propagated through full YOLO and U-Net downstream training because each downstream ablation would require additional detector and segmenter training.

## Interpretation

The results support RGB/IR fusion for lightweight perception, but they also show that architectural complexity is not automatically beneficial. Average fusion consistently improves over RGB-only and IR-only in both detection and segmentation. CM-JEPA gate-blend produces useful object-level saliency for detection, but segmentation requires sharper class boundaries and fine spatial structure than the proposed gate-blend preserves.

The fusion-quality metrics also show a limitation of classical image-level evaluation. A7 and average fusion have very similar entropy, spatial frequency, average gradient, and normalized mutual information, yet their segmentation mIoU differs substantially. For this reason, this project treats image-level fusion metrics as diagnostics rather than substitutes for downstream perception benchmarks.

## Repository Contents

```text
.
├── README.md
├── main.tex
├── main.pdf
├── references.bib
├── figures/
├── downstream_results.csv
├── fusion_ablation_metrics.csv
├── method_level_metrics_for_correlation.csv
├── reproducibility/
│   └── CM_JEPA_Erol_Kuluslu_Final.ipynb
└── source_material/
    └── project_proposal.pdf
```

The compiled PDF is included for convenient reading. LaTeX build artifacts such as `.aux`, `.log`, `.bbl`, and `.blg` are intentionally ignored.

## Reproducibility

The main executable artifact is:

```text
reproducibility/CM_JEPA_Erol_Kuluslu_Final.ipynb
```

The notebook is organized as a full research pipeline:

- dataset overview and alignment diagnostics
- environment and configuration setup
- RGB/IR manifest creation
- preprocessing and synthetic degradation
- RGB-only, IR-only, average, early-integration, and Laplacian baselines
- CM-JEPA fusion architecture and losses
- fusion training, export, and qualitative inspection
- YOLO11n detection evaluation
- U-Net/MobileNetV2 segmentation evaluation
- ablation, correlation analysis, and report artifact generation

Two execution modes are documented in the notebook:

- `SMOKE_TEST = True`: runs a small synthetic sanity-check pipeline.
- `SMOKE_TEST = False`: runs the real RGB/IR datasets under the configured `DATA_ROOT`.

## Paper Build

The IEEE-style report source is provided in `main.tex`, with references in `references.bib`.

To rebuild the paper:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If `bibtex` is unavailable but `bibtex8` is installed:

```bash
pdflatex main.tex
bibtex8 main
pdflatex main.tex
pdflatex main.tex
```

The `figures/` directory contains the exported plots and qualitative samples referenced by the report.

## Key Limitations

- A3-A6 ablation variants were evaluated with image-level fusion metrics only, not full downstream YOLO/U-Net training.
- The metric-correlation analysis uses only four downstream-evaluated methods, so it is descriptive rather than statistically conclusive.
- The experiments assume aligned RGB/IR pairs; sensor misalignment would require additional registration or robustness mechanisms.
- The downstream models receive a three-channel fused image for compatibility. A detector or segmenter designed for feature-level RGB/IR fusion may exploit cross-modal structure more directly.

## Citation

If you use this project as a reference, please cite the repository and the included IEEE-style report:

```bibtex
@misc{kulusslu2026cmjepa,
  author = {Erol Kuluslu},
  title = {Cross-Modal JEPA Gate-Blend Fusion for Lightweight RGB/Infrared Detection and Semantic Segmentation},
  year = {2026},
  howpublished = {GitHub repository}
}
```

## Author

Erol Kuluslu  
Department of Electrical-Electronics and Computer Engineering  
Abdullah Gul University, Kayseri, Turkey
