# Cross-Modal JEPA Gate-Blend Fusion for Lightweight RGB/Infrared Perception

This repository contains the final IEEE-style research package for **Cross-Modal JEPA Gate-Blend Fusion for Lightweight RGB/Infrared Detection and Semantic Segmentation**. The project studies whether paired visible RGB and thermal infrared images can improve lightweight downstream perception under difficult visual conditions such as night, fog, rain, low light, and reduced contrast.

The central empirical conclusion is intentionally conservative: **RGB/IR fusion is beneficial, but simple average fusion is the strongest overall representation in the verified experiments**. The proposed CM-JEPA gate-blend representation is competitive for strict object-detection localization, but it does not preserve dense semantic boundaries as effectively as average fusion for segmentation.

## Research Question

Can a cross-modal attention fusion model, strengthened with patch/token-level JEPA pretraining, improve lightweight object detection and semantic segmentation compared with RGB-only, IR-only, and average-fusion baselines?

## Main Contributions

- Built a reproducible RGB/IR fusion evaluation pipeline over 6025 aligned image pairs from M3FD, MSRS, and RoadScene.
- Evaluated RGB-only, IR-only, average fusion, learned attention ablations, and the final CM-JEPA gate-blend representation.
- Used downstream task metrics rather than only image-level fusion metrics: YOLO11n for detection and U-Net/MobileNetV2 for segmentation.
- Corrected the A7 evaluation protocol so CM-JEPA fusion metrics, YOLO inputs, and U-Net inputs all use the same gate-guided RGB/IR blend.
- Reported the negative result transparently: average fusion outperformed the proposed learned method overall.

## Final Downstream Results

| Method | mAP@0.5 | mAP@0.5:0.95 | mIoU |
|---|---:|---:|---:|
| RGB-only | 0.7616 | 0.4887 | 0.5307 |
| IR-only | 0.7064 | 0.4649 | 0.4976 |
| Average fusion | **0.7718** | **0.5115** | **0.5551** |
| CM-JEPA gate-blend | 0.7567 | 0.5080 | 0.4774 |

CM-JEPA gate-blend nearly matches average fusion for mAP@0.5:0.95, but the segmentation gap indicates loss of boundary and texture information.

## Dataset Layout

Raw datasets are not included. The notebook expects data under:

```text
/content/data/rgb_ir_datasets/
├── M3FD/
├── MSRS/
└── RoadScene/
```

The notebook standardizes dataset-specific folders into a common structure:

```text
DATA_ROOT/
├── M3FD/
│   ├── visible/
│   ├── infrared/
│   ├── labels/          # original labels when present
│   └── labels_yolo/     # converted YOLO txt labels
├── MSRS/
│   ├── visible/ or train/vi/
│   ├── infrared/ or train/ir/
│   └── masks/ or train/label/
└── RoadScene/
    ├── visible/
    └── infrared/
```

The verified manifest contains:

| Dataset | Role | Pairs |
|---|---|---:|
| M3FD | YOLO11n object detection | 4280 |
| MSRS | U-Net/MobileNetV2 segmentation | 1524 |
| RoadScene | fusion-quality and qualitative analysis | 221 |

Overall split: 4215 train, 902 validation, 908 test.

## Dependencies

The notebook installs missing packages automatically, but the equivalent environment commands are:

```bash
pip install torch torchvision opencv-python pillow numpy pandas matplotlib
pip install scikit-learn scikit-image tqdm ultralytics
pip install segmentation-models-pytorch albumentations kornia gdown
```

The verified full run was designed for Google Colab with GPU acceleration. A smoke-test mode is included for local or quick validation.

## Dataset Download Commands

The notebook contains these commands internally. They are listed here so the expected data acquisition path is explicit.

```bash
gdown --folder https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6 \
  -O /content/data/rgb_ir_datasets/M3FD --remaining-ok

git clone --depth 1 https://github.com/Linfeng-Tang/MSRS.git \
  /content/data/rgb_ir_datasets/MSRS

git clone --depth 1 https://github.com/jiayi-ma/RoadScene.git \
  /content/data/rgb_ir_datasets/RoadScene
```

If the M3FD Google Drive folder is unavailable because of access or quota restrictions, manually place `M3FD_Detection.zip` and `M3FD_Fusion.zip` under:

```text
/content/data/rgb_ir_datasets/M3FD/
```

Then rerun the dataset standardization cells.

## Reproduction Procedure

Main executable artifact:

```text
reproducibility/CM_JEPA_Erol_Kuluslu_Final.ipynb
```

Recommended execution order:

1. Run the environment and configuration cells.
2. For a quick check, set `SMOKE_TEST = True` in `ProjectConfig` and run all modules.
3. For the verified real-data pipeline, set `SMOKE_TEST = False`.
4. Run dataset download, extraction, standardization, and manifest creation.
5. Run CM-JEPA training or load the saved checkpoint:

```python
SKIP_JEPA_TRAINING = False   # train CM-JEPA from scratch
SKIP_JEPA_TRAINING = True    # load best_cm_jepa.pt if already available
```

6. Keep the final export setting enabled:

```python
USE_GATE_GUIDED_FUSION_EXPORT = True
```

7. Run the YOLO11n detection module.
8. Run the U-Net/MobileNetV2 segmentation module.
9. Run ablation, A7 repair, correlation, and final artifact-generation modules.

Important experiment-control defaults:

```python
FORCE_RETRAIN_BASELINE_YOLO = False
TRAIN_MISSING_YOLO = True
FORCE_RETRAIN_PROPOSED_GATE_YOLO = False
TRAIN_MISSING_UNET = True
TRAIN_MISSING_ABLATION = True
```

## Model Size and Diagnostics

The notebook records the following model-size information:

| Component | Total parameters | Trainable parameters |
|---|---:|---:|
| Fusion image model | 10.90M | 10.90M |
| CM-JEPA wrapper | 15.99M | 12.75M |

Additional saved diagnostics:

- `best_cm_jepa.pt` checkpoint size: 61.1 MB.
- CM-JEPA history rows: 30 epochs.
- Fusion feature tensor: `512 x 32 x 32` for `512 x 512` inputs.
- The raw decoder output can collapse visually; the final A7 representation therefore uses gate-guided RGB/IR blending.

Controlled latency was not benchmarked, so this package does not make a runtime claim.

## Repository Contents

```text
.
├── README.md
├── REVISION_NOTES.md
├── main.tex
├── main.pdf
├── references.bib
├── downstream_results.csv
├── fusion_ablation_metrics.csv
├── method_level_metrics_for_correlation.csv
├── figures/
├── reproducibility/
│   └── CM_JEPA_Erol_Kuluslu_Final.ipynb
└── source_material/
    └── project_proposal.pdf
```

## Paper Build

To rebuild the IEEE-style PDF:

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

## Known Limitations

- A3-A6 ablations were evaluated with fusion-quality metrics only, not full YOLO/U-Net downstream training.
- DenseFuse, CDDFuse, and other learned SOTA methods were not executed through the same downstream protocol; they are discussed as contextual references, not reported as numeric baselines.
- The metric-task correlation uses only four downstream-evaluated methods, so it is descriptive rather than statistically conclusive.
- Final aggregate gate statistics were not exported; future work should log gate mean, variance, entropy, and saturation rates.
- Segmentation failure cases should be expanded with prediction-mask overlays in a future revision.

## Author

Erol Külüşlü  
Department of Electrical-Electronics and Computer Engineering  
Abdullah Gül University, Kayseri, Turkey
