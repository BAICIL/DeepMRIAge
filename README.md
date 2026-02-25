# HBS-AgeNet (HeadBrainSkull-AgeNet)

Predict chronological age from structural 3D T1-weighted MRI using three complementary inputs: Head, Brain, and Skull.

Author: Javad Sohankar (j.sohankar@bannerhealth.com)
Affiliation: Banner Alzheimer’s Institute — Phoenix, Arizona, USA

---

## Overview

HBS-AgeNet provides three deep-learning models (ResNet-18–style CNNs) trained to predict chronological age from structural T1 MRI scans:

- HeadAge — Head images (includes skull + facial features)
- BrainAge — Brain-only images
- SkullAge — Skull-only images (derived from head + brain images)

Each model must be used only with the image type it was trained on (e.g., use HeadAge with head images). Using the wrong image type will produce inconsistent results.

---

## Training and Evaluation Data

Models were trained on five datasets (total N=7,932):

- ABIDE
- ICBM
- IXI
- NACC
- OASIS

They were evaluated on:

- ADNI (public), N=11,007
- APOE 1.0 & APOE 2.0 (private), N=2,100

Note: Ten sample scans from IXI dataset is included for testing the model, the entire dataset can be downloaded at: https://brain-development.org/ixi-dataset/
---

## Preprocessing (FreeSurfer)

Raw T1 scans were processed with FreeSurfer 7. Model inputs are based on the following FreeSurfer outputs:

- HeadAge: nu.mgz
- BrainAge: brain.mgz
- SkullAge: computed via Extract_Skull.py using:
  - nu.mgz (head)
  - brain.mgz (brain mask)

Skull extraction saves a skull-only volume to disk, which is then used for SkullAge training/testing.

---

## Installation

This package was implemented and tested with Python 3.11.

Option A — Install directly from GitHub (recommended)
```
pip install git+https://github.com/<ORG_OR_USER>/<REPO>.git
```
Option B — Install from requirements.txt
```
pip install -r requirements.txt
```
Option C — Install via setup.py
```
python3 setup.py
```
Tip: Use a virtual environment (venv/conda) to avoid dependency conflicts.

---

## Quick Start (Run on Included IXI Sample Lists)

The commands below run inference on the included sample CSV input lists (IXI).
Each command selects the corresponding model and writes prediction results to the output folder.
```
python3 -u AgePredication_Final.py \
  --inList "IXI_Head_Scan_Age--Input_List.csv" \
  --model "HeadAge" \
  --ext "nii.gz" \
  --out "IXI_Head_Scan_Age_predication_results"
```
```
python3 -u AgePredication_Final.py \
  --inList "IXI_Brain_Scan_Age--Input_List.csv" \
  --model "BrainAge" \
  --ext "nii.gz" \
  --out "IXI_Brain_Scan_Age_predication_results"
```
```
python3 -u AgePredication_Final.py \
  --inList "IXI_Skull_Scan_Age--Input_List.csv" \
  --model "SkullAge" \
  --ext "nii.gz" \
  --out "IXI_Skull_Scan_Age_predication_results"
```
Notes on inputs:
- The model accepts four type of scan extensions (np, nii, nii.gz, mgz), but in each run use only one type.
- The input CSV format is demonstrated in the included IXI_*_Input_List.csv files.
- --ext should match your scan file extension (e.g., nii.gz).

---

## Skull Extraction (for SkullAge)

To generate skull-only volumes for your own data:
```
python3 Extract_Skull.py --head nu.mgz --mask brain.mgz --out result
```
Arguments:
- --head: path to nu.mgz (FreeSurfer output)
- --mask: path to brain.mgz (FreeSurfer output)
- --out: name prefix for saving the extracted skull volume

---

## Troubleshooting

- Wrong model / wrong image type: results may look unstable or incorrect. Double-check --model matches your data type.
- FreeSurfer files missing: HeadAge/BrainAge/SkullAge pipelines assume FreeSurfer outputs exist and are correctly generated.

---

## Citation

If you use this repository in academic work, please cite the paper (and github repository):

### Paper

APA:

Li, Y., Sohankar, J., Luo, J., Li, J., & Su, Y. (2025). BrainPath: A Biologically-Informed AI Framework for Individualized Aging Brain Generation. arXiv preprint arXiv:2508.16667.

Bibtex:

@article{li2025brainpath,
  title={BrainPath: A Biologically-Informed AI Framework for Individualized Aging Brain Generation},
  author={Li, Yifan and Sohankar, Javad and Luo, Ji and Li, Jing and Su, Yi},
  journal={arXiv preprint arXiv:2508.16667},
  year={2025}
}

### Gihub Repository

Sohankar, J. HBS-AgeNet (HeadBrainSkull-AgeNet): Predicting chronological age from T1-weighted MRI using head, brain, and skull models. (GitHub repository).

---

## Contact

For questions, collaborations, or issues:
- Javad Sohankar — j.sohankar@bannerhealth.com
