# Heart Disease MLOps Assignment

This repository implements the BITS Pilani AIMLCZG523 MLOps Assignment 01.

## Execution order

1. Open `notebook/MLOps_Assignment_01_End_to_End.ipynb`.
2. Run the numbered notebook cells in order.
3. The notebook creates the dataset, EDA artifacts, model, MLflow run, and model package.
4. Run `pytest -q`.
5. Start the API with `uvicorn src.api:app --host 0.0.0.0 --port 8000`.
6. Build the Docker image.
7. Deploy with the Kubernetes manifest if Minikube or another Kubernetes environment is available.

## API endpoints

- `/health`
- `/predict`
- `/metrics`
- `/docs`

## Docker

```bash
docker build -t heart-disease-mlops:latest .
docker run --rm -p 8000:8000 heart-disease-mlops:latest
```

## Kubernetes

```bash
minikube start
minikube image load heart-disease-mlops:latest
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get services
minikube service heart-disease-api-service --url
```
