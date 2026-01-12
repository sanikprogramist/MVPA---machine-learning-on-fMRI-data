# mvpa — Machine Learning from fMRI

This repository contains a machine learning pipeline that decodes gaze direction from fMRI brain activity recorded while participants explored a virtual environment. Using multi-voxel pattern analysis (MVPA) and searchlight-style models, the pipeline predicts the direction a person was looking from their brain responses.

Important: This work requires raw fMRI/behavioral data that cannot be shared publicly due to participant privacy. The code can be run with your own (appropriately consented and preprocessed) datasets.

The core analyses train and evaluate classifiers/regressors on brain activity features (beta images), use cross-validation, and report held-out decoding performance — this is supervised machine learning applied to neuroimaging.

Project layout
- `main.py` — Main analysis pipeline . It orchestrates preprocessing, beta image generation, model training, cross-validation, and results aggregation.
- `compute_beta_images.py` — Convert first-level GLM results into beta images used as features for decoding.
- `searchlight_on_raw_images.py` — Legacy/experimental script that ran searchlights directly on raw images; the current pipeline uses 
- `file_path_manager.py` — Centralize dataset and output paths (edit before running).
- `requirementsMVPA.txt` — Python dependencies.

Results example
![Decoding results example](results.jpg)

Quick start
1. Prepare environment and install dependencies:

```powershell
cd C:\Users\Stimulus\Documents\mvpa
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirementsMVPA.txt
```

2. Configure paths: update `file_path_manager.py` to point to your local fMRI data and output directory.

3. Run the main analysis:

```powershell
python main.py
```

Notes & data privacy
- You must run this with your own fMRI data; no subject data is included in this repository.
- Ensure subjects have consented to data sharing and that you follow your institution's data-use policies.

License
This repository is available under the MIT License (see LICENSE).
