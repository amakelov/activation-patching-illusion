# Is this the subspace you are looking for? An Interpretability Illusion for Subspace Activation Patching

This repository contains code to reproduce results from the paper "An
Interpretability Illusion for Subspace Activation Patching".

## Indirect Object Identification (IOI) task (sections 4 and 5)
Description of relevant files:
- `data_utils.py`: tools for working with the IOI dataset
- `model_utils.py`: tools to intervene on transformerlens models and train DAS subspaces
- `ioi_interventions.ipynb`: notebook to train DAS and related interventions
- `ioi_analysis.ipynb`: notebook to analyze IOI interventions (using already
  trained/computed subspaces saved as files in this repository)

## Factual recall (section 6)
Description of relevant files:
- `fact_utils.py`: tools to download necessary datasets
- `fact_patching.ipynb`: code for fact patching experiments in
  sections 6.1. and 6.4. of the paper
- `fact_patching_plots.ipynb`: notebook to recreate factual recall plots for
  sections 6.1 and 6.4. of the paper
- `fact_editing.ipynb`: notebook to run fact editing experiments from section
  6.3. of the paper
- `fact_editing_plots.ipynb`: notebook to recreate ROME-to-subspace-intervention
  plots from section 6.3

## Experiments to validate our model of the illusion (section 7)
- `theory_experiments.ipynb`: experiments for singular values of MLP weights and
  evaluating distortion introduced by GELU nonlinearity