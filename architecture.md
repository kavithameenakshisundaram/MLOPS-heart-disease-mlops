# Architecture

```text
UCI Heart Disease Dataset
          |
          v
Data Acquisition
          |
          v
EDA and Data Cleaning
          |
          v
Feature Engineering
          |
          v
+---------------------------+
| Logistic Regression       |
| Random Forest             |
+---------------------------+
          |
          v
Cross Validation
          |
          v
Model Selection and Tuning
          |
          v
MLflow Experiment Tracking
          |
          v
Reproducible Model Pipeline
          |
          v
FastAPI Prediction Service
          |
          v
Docker Container
          |
          v
Kubernetes Deployment
          |
          v
Logging and Prometheus Metrics
```
