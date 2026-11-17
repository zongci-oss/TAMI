# TaMI: Target-Mask-Informed Imputation for Time Series

TaMI (Target-Mask-Informed Time-Series Imputation) is an **offline batch-repair** framework for multivariate time-series imputation. It studies
*transductive mask-only target adaptation*: the observation-mask collection of an incomplete target
batch is available before repair, while the target values withheld for evaluation remain inaccessible.
TaMI uses that target-mask information to configure the training corruption process, but does not
use the hidden target values during mask modeling or imputation-model training.

## Table of Contents
- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Setup & Dependencies](#setup--dependencies)
- [Datasets](#datasets)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Scope of the Reported Results](#scope-of-the-reported-results)

## Overview

TaMI consists of two coordinated modules:

- **Missingness Distribution Alignment (MDA)** — models the target-mask collection. It maps a binary
  observation mask into a continuous structural mask representation (CMR), fits a latent mask
  generator (Mask VAE) on that representation, then samples from the decoder and calibrates a
  sparsity threshold so the generated training masks reproduce the target missing rate. The output is
  the training mask `Omega_tr` consumed by SDR.
- **Structural Dependency Recovery (SDR)** — the reconstruction backbone. It reorganizes the
  multivariate series into variable-wise system tokens for cross-variable dependency modeling, then
  refines temporal consistency with a stateful evolution operator (Mamba). It is trained under the
  MDA-generated masks with the objective `L_SDR = L_miss + alpha * L_obs`, where `L_miss` acts on
  withheld entries and `L_obs` enforces consistency on observed entries.

The two modules are decoupled and SDR can be trained independently of MDA.

## Repository Layout

| Path | Role |
| --- | --- |
| `MDA.py` | Missingness Distribution Alignment: CMR, Mask VAE, training-mask generation. Produces `mda_model.pth`. |
| `TaMI.py` | Structural Dependency Recovery: the `SDR` backbone, the `TaMI` imputation model, and `SDR_loss`. |
| `iTransformer/` | Cross-variable attention + Mamba backbone components. |
| `main.py` | Training / evaluation / imputation entry point. |
| `arguments.py` | Hyperparameters and per-dataset configurations (`TaMI_arguments()`). |
| `dataloader.py` | Dataset loading; `--model_type TaMI` selects the TaMI batching path. |
| `dataset_generating_scripts/` | Raw-data download and preprocessing. |
| `SavedModel/TaMI/`, `logs/TaMI/` | Checkpoints and logs (created at runtime). |

## Setup & Dependencies

Python: `3.10`

`requirements.txt` (put these lines exactly):

```
h5py==3.12.1
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
six==1.16.0
torch==2.3.1
tqdm==4.66.2
tsdb==0.6.2
packaging==24.1
transformers==4.44.2
mamba-ssm==2.2.2
```

Create and activate an environment (pick one):

Conda:
- `conda create -n tami python=3.10 -y`
- `conda activate tami`

Venv:
- `python3.10 -m venv .venv`
- `source .venv/bin/activate` (Windows: `.\venv\Scripts\activate`)

Install dependencies:
- Save the pinned list above to `requirements.txt` at the repo root.
- `pip install -r requirements.txt`

## Datasets

The project supports common multivariate time-series datasets and preprocessed `.h5` files:
Air Quality (Beijing), Electricity, and ETT(m1). Reported conditions use a controlled
structured-mask family together with random missingness.

## Dataset Preparation

1) Download the original datasets:
- `bash data_downloading.sh`

2) Process the datasets:
- `cd dataset_generating_scripts`
- Open `gene_UCI_BeijingAirQuality_dataset.py` and set the missingness mode:
\ \ \ \ Set `missing_mode` from choices=\"random\", \"structural\".
- Generate processed data: `bash dataset_generating.sh`

3) Rename the processed `.h5` file:
- **Enter** `data` and **rename the current** `.h5` **to** `datasets.h5` (to make the training mode unambiguous).

4) Run MDA to learn the target mask distribution and write the training masks:
- `python MDA.py` -> produces `mda_model.pth`

5) Run the main pipeline:
- `python main.py`

## Training

- Default entry point: `main.py`. Hyperparameters are defined in `arguments.py`, e.g. `model=TaMI`,
  device, learning rate, batch size, and the `SDR_loss` weighting `alpha`.
- `main.py` loads `mda_model.pth` through `MDA.generate_masks` to build the training masks, then fits
  the `TaMI` model with `SDR_loss`.
- Logs and checkpoints are written to `logs/TaMI/<dataset>/` and `SavedModel/TaMI/<dataset>/`.
- Example: `python main.py --model TaMI`

## Evaluation

- Open `arguments.py`, find the model stage parameter (e.g., `default='train', help="Model stage"`) and
  change `default='train'` to `default='test'`.
- Save the file, then run: `python main.py`.
- The script evaluates on the test set and reports MAE / RMSE / MRE in logs and console.
- To impute only, set the stage to `'impute'`.

Note: in the offline protocol the target masks are revealed before fitting and are treated as
transductive target-side metadata. The target values hidden by those masks are used only for metric
computation, never for fitting or model selection.

## Scope of the Reported Results

The experiments evaluate condition-specific adaptation under the stated offline adaptation protocol,
comparing against non-adaptive reference configurations that do not receive the aggregate target-side
information. They do not establish generalization to previously unseen missingness mechanisms, and the
aggregate missing rate alone should not be read as evidence that generated masks reproduce the full
joint target-mask distribution.

## Citation

If you use this code, please cite the TaMI paper.
