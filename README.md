# CPSC552 Final Project

This repository contains the codebase and results for our final project in CPSC 552, focused on building a multimodal survival prediction model using RNA-seq data, clinical variables, and whole slide images (WSIs).

## Project Structure

- `code/`: Contains the main model script (`CPSC452Model.py`) and related utilities, including data cleaning and image downloading. 
- `results_images/`: Stores result visualizations (Validation C-index plot, Kaplan–Meier curves, risk scatter plot).
- `workflow.png`: Visual representation of the project pipeline. 

## Input Data

All input data used in model training are stored on the Yale HPC cluster. The relevant data paths are:

```python
RNASEQ_PATH = "/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/filtered_rnaseq_data.npy"
RNA_IDS     = "/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/filtered_rna_ids.npy"
CLIN_CSV    = "/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/cleaned_clinical.csv"

WSI_DIRS = [
    "/gpfs/gibbs/project/cpsc452/cpsc452_jl4286/image",
    "/gpfs/gibbs/project/cpsc452/cpsc452_am4289/image",
    "/gpfs/gibbs/project/cpsc452/cpsc452_yg427/data/image_svs_manual",
    "/gpfs/gibbs/project/cpsc452/cpsc452_yy743/image_svs_manual",
]
```

## How to Run
### Set up your environment 
Create and activate the environment using the provided tgbm.tml file:

```python
conda env create -f tf_multimodal_cpu.yml
conda activate tf_multimodal_cpu
```

If the above code doesn't work, you can manually installed the dependencies required. 

Dependencies:
```python
  - python=3.10
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - torch=1.13.1
  - torchvision
  - openslide-python
  - lifelines
  - jupyter
  - notebook
  - ipykernel
```

### Train the model
Execute the training script:

```python
python code/CPSC452Model.py
```
