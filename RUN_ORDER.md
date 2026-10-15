# Recommended Run Order in the BITS Lab

## Phase 1: Environment

Open Jupyter and create or upload the notebook.

Run notebook cells 1 through 18.

## Phase 2: Machine Learning

Run cells 19 through 27.

At this point you should have:
- EDA figures
- Cross-validation results
- Final test metrics
- Final model

## Phase 3: MLOps

Run cells 28 through 36.

At this point you should have:
- MLflow experiment
- Saved model
- Metadata
- requirements.txt

## Phase 4: Source files

Use:
- src/download_dataset.py
- src/train.py
- src/api.py
- tests/test_model.py

## Phase 5: CI/CD

Push the repository to GitHub and verify `.github/workflows/ci.yml`.

## Phase 6: Docker

Build and run the Docker image.

## Phase 7: Kubernetes

Only perform this phase after confirming that Minikube or Kubernetes is available in the BITS lab.

## Phase 8: Monitoring

Open `/metrics` and capture API logs.

## Phase 9: Documentation

Capture screenshots and prepare the report.
