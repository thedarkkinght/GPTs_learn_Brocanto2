# GPTs Learn Brocanto2

Core analysis code for the manuscript:

**Predictive and feedback signals differently shape the formation of group-level and individualized language representations**

This repository contains the lightweight code, stimulus files, model-derived representations, and statistical maps used for the main analyses. The full behavioral and neuroimaging dataset includes 102 participants and is approximately 2 TB, so it is not included in this GitHub repository.

## Overview

The study used a seven-day Brocanto2 artificial-language learning paradigm. Participants performed grammaticality-judgment learning with corrective feedback, with fMRI acquired during early learning. Matched GPT-style models were trained with different learning objectives:

- **GPT-P**: next-token prediction
- **GPT-F**: feedback-based grammaticality judgment
- **GPT-PF**: combined prediction and feedback objectives

The notebooks reproduce the main computational analyses, including model training, model-brain representational similarity analysis (RSA), and individualized neurocomputational modeling.

## Repository Contents

- `0_train_prediction_model.ipynb`  
  Train the prediction-based GPT model.

- `1_train_feedback_model.ipynb`  
  Train the feedback-based grammaticality-judgment GPT model.

- `2_train_pf_model.ipynb`  
  Train the combined prediction-feedback GPT model.

- `3_rsa.ipynb`  
  Run RSA analyses between model representations and fMRI-derived neural representations.

- `4_personalized_model.ipynb`  
  Train individualized models using early behavioral and neural information to predict Day 7 generalization.

- `MATERIAL.xlsx`  
  Brocanto2 stimulus metadata.

- `text/`  
  Text corpora used for model training and tokenization.

- `tokenise.py`, `tokenized_data/`  
  BPE tokenizer code and saved tokenizer files.

- `right_corpus.py`  
  Helper file defining Brocanto2 construction subsets.

- `numpy_array/rsa/`  
  Precomputed model representation arrays used in RSA.

- `tfce_maps/`  
  Threshold-free cluster enhancement statistical maps for the reported RSA results.

- `requirements.txt`  
  Python package list used for the analyses.

## Data Availability

The full behavioral and neuroimaging data are not distributed in this repository because of their size and participant-data restrictions. The full dataset will be available upon reasonable request from the corresponding author.

Some notebooks, especially `3_rsa.ipynb` and `4_personalized_model.ipynb`, require restricted behavioral or neural files that are not included here, such as participant-level fMRI response maps and behavioral output files. The included `.npy` files and TFCE maps provide lightweight derived materials for inspecting and reproducing the core model-representation and group-level RSA workflow.

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The analyses were developed with PyTorch, Hugging Face Transformers, scikit-learn, Nilearn, MNE-RSA, ANTsPy, NumPy, Pandas, and related scientific Python packages. GPU acceleration is recommended for model training.

## Suggested Workflow

Run the notebooks in numerical order:

1. `0_train_prediction_model.ipynb`
2. `1_train_feedback_model.ipynb`
3. `2_train_pf_model.ipynb`
4. `3_rsa.ipynb`
5. `4_personalized_model.ipynb`

The first three notebooks train the computational models. The RSA and individualized modeling notebooks depend on preprocessed behavioral/neural data and derived model representations.

