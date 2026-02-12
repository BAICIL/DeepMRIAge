# Predicting Chronological Age based on 3D Brain MRI Images 
### Javad Sohankar (j.sohankar@bannerhealth.com)
### Banner Alzheimer's Institute, Phoenix, Arizona, US

We trained three deep learning models (based on ResNet18 architecture) to predict chronological age from structural T1-weighted MRI scans. The three models (HeadAge, BrainAge, SkullAge) were trained based on three different types of MRI image: 1) Head (including skull and facial features), 2) Brain only, 3) Skull only. 

Training process used data from five data sets (ABIDE, ICBM, IXI, NACC, OASIS; N=7932), and then evaluated on a separate dataset (ADNI, N=11007).

First, raw T1-MRI scans were processed using FreeSurfer 7 software. For HeadAge model, “nu.mgz" files were used as head image for training and afterward evaluating it. Similarly, BrainAge model used “brain.mgz” files for training/evaluation.  For SkullAge model, we started with extracting the skull portion based on "nu.mgz" (head image) and "brain.mgz" (brain image), and saved them to disk. Afterwards these skull images were used for training and testing the SkullAge model.

## Installation
We have implemented and tested the model on python 3.11. For installing dependencies, you can choose between these three methods:

1) Directly from GitHub using `pip` (which utilizes the pyproject.toml file):

```bash
pip install git+https://github.com
```

2) Use the "requirements.txt" file, and install dependencies:
```
pip install -r requirements.txt
```

2) run the "setup.py" script which fetches the libraries from the "requirements.txt" file and installs them.
```
python3 setup.py
```

## Running The Package  
The following commands will run the models on the included sample data from IXI dataset. The first command uses HeadAge model for predicting age from head images, the second one utilizes BrainAge model for predicting age from brain images, and the last one uses SkullAge model on skull images. **Keep in mind each model should be used only with the image type it was trained on, otherwise it will lead to inconsistent results**
```
python3 -u AgePredication_Final.py --inList "IXI_Head_Scan_Age--Input_List.csv"  --model "HeadAge" --ext "nii.gz" --out "IXI_Head_Scan_Age_predication_results"
python3 -u AgePredication_Final.py --inList "IXI_Brain_Scan_Age--Input_List.csv"  --model "BrainAge" --ext "nii.gz" --out "IXI_Brain_Scan_Age_predication_results"
python3 -u AgePredication_Final.py --inList "IXI_Skull_Scan_Age--Input_List.csv"  --model "SkullAge" --ext "nii.gz" --out "IXI_Skull_Scan_Age_predication_results"
```
