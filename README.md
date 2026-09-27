# GeoSense Agent — Phase 1 & Phase 2

GeoSense Agent is a geospatial AI project for identifying promising locations using geospatial features, satellite imagery, and machine-learning/deep-learning models.

---

# Phase 1 Status

Phase 1 implements the core geospatial data pipeline, feature engineering, machine-learning training, SHAP explainability, site scoring, and automated tests.

## Phase 1 Project Structure

```text
GeoSense_Agent/
├── data/
│   ├── processed/
│   └── shapefiles/
├── models/
│   └── saved/
├── notebooks/
├── outputs/
│   └── reports/
├── src/
│   └── phase1_ml/
└── tests/
```

### Phase 1 Results

- Candidate locations: **408**
- Good sites: **130**
- Bad sites: **278**
- Random Forest test accuracy: **96.34%**
- XGBoost test accuracy: **96.34%**
- SHAP explainability completed
- Site scorer implemented
- Automated tests: **4 passed**

The Phase 1 labels were generated from the available geospatial features. The flood-risk feature was `0.0` for all candidates because no populated flood-zone table was available in the implemented dataset.

---

# Phase 2 Status — Deep Learning + Satellite Imagery

Phase 2 extends GeoSense with Sentinel-2 satellite imagery, semantic segmentation, foundation-model experiments, imagery classification, and change detection.

The Phase 2 laboratory practicum covers deep-learning foundations, satellite preprocessing, U-Net training, foundation-model fine-tuning/evaluation, and image classification/change detection.

> **Implementation note:** This README documents the workflow that was actually implemented. Where the laboratory manual specified a workflow that was not executed, the difference is stated explicitly rather than being presented as completed.

## Phase 2 Environment

- Python 3.12.14
- PyTorch 2.14.0+cpu
- TorchGeo 0.9.0
- TerraTorch 1.2.13
- timm 1.0.30
- Rasterio 1.5.1
- CPU execution
- Google Earth Engine project: `ee-makzaro134242354`

## Sentinel-2 Processing

The implemented workflow used:

```text
Collection: COPERNICUS/S2_SR_HARMONIZED
Period: January–December 2024
Cloud filter: <30%
Composite: median
```

Study area:

```text
80.199, 13.036, 80.316, 13.116
```

The Earth Engine test found **28 Sentinel-2 images**.

Primary six-band feature set:

```text
B2, B3, B4, B8, B11, B12
```

Computed indices:

```text
NDVI = (NIR - Red) / (NIR + Red)
NDWI = (Green - NIR) / (Green + NIR)
NDBI = (SWIR - NIR) / (SWIR + NIR)
```

Observed ranges:

| Index | Minimum | Maximum |
|---|---:|---:|
| NDVI | -0.314181 | 0.748792 |
| NDWI | -0.665641 | 0.497937 |
| NDBI | -0.307281 | 0.217485 |

The Sentinel-2 feature table contains **408 rows and 13 columns**, with no missing values.

## U-Net Segmentation

Five classes were used:

1. Urban
2. Vegetation
3. Water
4. Bare Land
5. Agriculture

The implemented 2024 pseudo-label raster contained:

| Class | Percentage |
|---|---:|
| Urban | 3.06% |
| Vegetation | 17.83% |
| Water | 11.01% |
| Bare Land | 35.82% |
| Agriculture | 32.27% |

A total of **28 valid 224×224 chips** were generated.

Model configuration:

```text
Encoder: ResNet34
Input channels: 6
Classes: 5
Loss: CrossEntropyLoss
Optimizer: Adam
Learning rate: 0.001
```

### U-Net validation

- Pixel accuracy: **75.74%**
- Mean IoU: **55.94%**

| Class | IoU |
|---|---:|
| Urban | 17.89% |
| Vegetation | 68.93% |
| Water | 78.80% |
| Bare Land | 58.23% |
| Agriculture | 55.84% |

The Urban class was the weakest class in this validation experiment.

## Prithvi Foundation Model

The implemented foundation model was:

```text
ibm-nasa-geospatial/Prithvi-EO-2.0-300M
```

This is an adaptation because the laboratory manual names a smaller Prithvi model.

The model contains:

- 24 transformer blocks
- 1024-dimensional embeddings
- six spectral input bands
- 224×224 spatial input

Four quarterly 2024 inputs were prepared:

```text
Q1 → chennai_prithvi_t1.tif
Q2 → chennai_prithvi_t2.tif
Q3 → chennai_prithvi_t3.tif
Q4 → chennai_prithvi_t4.tif
```

Six 224×224 spatial patches covered all 408 candidate locations.

### Fine-tuning adaptation

The actual Prithvi architecture uses transformer blocks rather than conventional CNN `layer1`/`layer2` modules. Therefore the implementation:

- froze patch embedding
- froze transformer blocks 0–21
- froze final normalization
- trained transformer blocks 22–23
- trained a segmentation head

Parameter summary:

```text
Total:       306,247,173
Trainable:    27,553,285
Frozen:      278,693,888
```

Optimization:

```text
Backbone LR: 1e-5
Head LR:     1e-4
Optimizer:   AdamW
Weight decay: 1e-4
Loss:        CrossEntropyLoss
```

Three CPU fine-tuning epochs completed.

| Epoch | Train Accuracy | Validation Accuracy | Validation Loss |
|---:|---:|---:|---:|
| 1 | 34.95% | 40.43% | 1.5697 |
| 2 | 42.45% | 54.86% | 1.1791 |
| 3 | 48.78% | 53.78% | 1.0988 |

## U-Net vs Prithvi

On the same five-chip validation split:

| Model | Pixel Accuracy | Mean IoU |
|---|---:|---:|
| U-Net | **75.74%** | **55.94%** |
| Adapted Prithvi | **53.78%** | **32.93%** |

These results apply to this particular small pseudo-labelled validation experiment and should not be generalized as an inherent model ranking.

## Prithvi Feature Experiment

Random Forest experiments produced:

| Feature set | Accuracy | Macro F1 |
|---|---:|---:|
| Phase 1 baseline | 96.34% | 95.63% |
| Prithvi only | 56.10% | 56.07% |
| Phase 1 + Prithvi | 97.56% | 97.24% |

Five-fold cross-validation:

| Feature set | Mean Accuracy | Std. Dev. |
|---|---:|---:|
| Phase 1 baseline | 99.76% | 0.49 pp |
| Prithvi only | 63.98% | 4.48 pp |
| Phase 1 + Prithvi | 97.56% | 2.04 pp |

The labels were constructed from Phase 1 variables, so the strong Phase 1 baseline is expected and is not independent real-world validation.

## Image Classifier

`src/phase2_dl/image_classifier.py`:

- prefers the fine-tuned Prithvi model
- falls back to U-Net
- extracts 224×224×6 patches
- fills missing values with band means
- computes NDVI and NDWI
- returns dominant class, distribution, and confidence

Test location:

```text
Latitude: 13.0827
Longitude: 80.2707
Radius: 500 m
```

Result:

```text
Dominant class: Bare Land
Confidence: 34.70%
Runtime: 6.284 seconds
```

Distribution:

```text
Urban          3.64%
Vegetation    24.02%
Water          8.22%
Bare Land     34.70%
Agriculture   29.41%
```

The laboratory target of under 3 seconds was **not achieved** on the CPU implementation.

## Change Detection

The manual specifies a 2015-versus-2023 comparison. The implemented project instead used the available quarterly 2024 imagery:

```text
Early: Q1 2024
Late:  Q4 2024
```

Results:

```text
Early NDVI mean:                0.0896
Late NDVI mean:                 0.1240
Mean absolute NDVI difference:  0.0411
Maximum absolute difference:    0.3082
Pixels above threshold:         3.59%
Threshold:                      0.15
Change flag:                    False
```

This is an adapted 2024 seasonal comparison, not a 2015–2023 historical result.

---

# Project Structure

```text
GeoSense_Agent/
├── data/
│   ├── processed/
│   └── shapefiles/
├── models/
│   └── saved/
├── notebooks/
├── outputs/
│   └── reports/
├── src/
│   ├── phase1_ml/
│   └── phase2_dl/
└── tests/
```

Important Phase 2 scripts include:

```text
src/phase2_dl/
├── sentinel2_test.py
├── sentinel2_features.py
├── prepare_unet_data.py
├── generate_labels.py
├── chip_images.py
├── train_unet.py
├── evaluate_unet.py
├── load_prithvi_official.py
├── extract_prithvi_patch_features.py
├── finetune_prithvi.py
├── evaluate_models.py
├── plot_model_comparison.py
└── image_classifier.py
```

Large generated datasets, model checkpoints, reports, plots, secrets, and caches are excluded from Git through `.gitignore`.

---

# Main Limitations

1. Execution was CPU-only.
2. The segmentation experiment used 28 chips: 23 training and 5 validation.
3. Labels were pseudo-labels rather than independent human annotations.
4. Prithvi-EO-2.0-300M was used instead of the smaller model named in the manual.
5. Transformer freezing was adapted to blocks 22–23.
6. The current segmentation adaptation uses six-channel spatial chips with a singleton temporal dimension.
7. The exact 2015/2023 imagery workflow was not executed.
8. The image-classifier runtime was 6.284 seconds, above the 3-second target.
9. Phase 1 labels are rule-derived and therefore do not constitute independent real-world ground truth.

---

# Git Version History

The Phase 2 source implementation was committed as:

```text
6dd28ea Implement GeoSense Phase 2 deep learning pipeline
```

Earlier Phase 1 commits include:

```text
d917844 Add Phase 1 health check
257f41c Add project dependency requirements
a3cb99e Add Phase 1 project documentation
a5f42b3 Add SHAP feature importance report
5755efa Implement GeoSense Phase 1 ML pipeline
```

Final Git verification showed:

```text
HEAD -> main
origin/main
nothing to commit, working tree clean
```

---

# Completion Summary

## Phase 1

- [x] Geospatial data pipeline
- [x] PostGIS setup and ingestion
- [x] Candidate grid
- [x] Feature engineering
- [x] Rule-based labels
- [x] Random Forest/XGBoost
- [x] SHAP explainability
- [x] Site scorer
- [x] Automated tests
- [x] GitHub version control

## Phase 2

- [x] Deep-learning environment
- [x] Earth Engine connection
- [x] Sentinel-2 2024 processing
- [x] Pseudo-labelling
- [x] 224×224 chips
- [x] U-Net training/evaluation
- [x] Prithvi-EO-2.0-300M loading
- [x] Prithvi feature extraction
- [x] Prithvi fine-tuning adaptation
- [x] Model comparison
- [x] Image classifier
- [x] Adapted 2024 change detection
- [x] GitHub synchronization
- [ ] Exact 2015/2023 workflow
- [ ] Under-3-second classifier target
- [ ] Independent human-labelled validation

## Conclusion

GeoSense Agent 2.0 now contains an end-to-end geospatial and satellite-imagery workflow extending the Phase 1 site-scoring system with deep learning and foundation-model experiments. The documented results and limitations distinguish the implemented system from the laboratory specification and identify the remaining work required for a closer reproduction of the requested workflow.
