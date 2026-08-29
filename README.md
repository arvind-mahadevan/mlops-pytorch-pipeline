# mlops-pytorch-pipeline

A production-style ML pipeline that trains a ResNet-18 image classifier on CIFAR-10 and serves predictions via a REST API — built with Docker for containerization and Kubernetes for orchestration.

## Project structure

├── src/                    # Application code
│   ├── model.py             # ResNet-18 classifier definition
│   ├── dataset.py            # CIFAR-10 loading + transforms
│   ├── train.py              # Training loop with early stopping
│   └── serve.py               # FastAPI inference server
│
├── configs/
│   └── training_config.yaml   # Hyperparameters, paths (read by train.py)
│
├── docker/
│   ├── Dockerfile.train       # Multi-stage training image
│   └── Dockerfile.serve       # Multi-stage serving image (non-root, healthcheck)
│
├── k8s/
│   ├── namespace.yaml          # ml-training namespace
│   ├── configmap.yaml          # Mounts training_config.yaml into the Job
│   ├── pvc.yaml                # Persistent storage for data + checkpoints
│   ├── training-job.yaml       # Runs training as a batch Job
│   ├── serving-deployment.yaml # 2-replica model server with health probes
│   ├── serving-service.yaml    # ClusterIP exposing the model server
│   └── hpa.yaml                 # Autoscaling for the serving deployment
│
├── requirements/
│   ├── train.txt              # torch, torchvision, pyyaml (CPU-only wheels)
│   └── serve.txt               # fastapi, uvicorn, torch, pillow
│
└── tests/
    └── test_model.py           # Unit tests for model.py

## Setup — local training

```bash
pip install -r requirements/train.txt
python src/train.py
```

Reads config from `configs/training_config.yaml`. Trains a ResNet-18 on CIFAR-10 (auto-downloaded on first run), logs metrics as JSON lines, saves the best checkpoint to `checkpoints/classifier_v1.pt`, and stops early if validation loss stops improving.

## Setup — local serving

```bash
pip install -r requirements/serve.txt
cd src
uvicorn serve:app --host 0.0.0.0 --port 8080
```

- `GET /health` — readiness check
- `POST /predict` — accepts an image file, returns per-class probabilities

## Setup — Docker

```bash
# Training
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# Serving
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Setup — Kubernetes

Requires a running cluster (Minikube, kind, or cloud-managed) with `kubectl` configured.

```bash
# Point Docker at the cluster's registry (Minikube example) and build images
eval $(minikube docker-env)
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Deploy training
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml
kubectl get pods -n ml-training -w

# Deploy serving once training completes
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# Test
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Notes

- CPU/memory resource requests in the manifests are set conservatively to run in resource-constrained development clusters; scale up for production.
- GPU support (`nvidia.com/gpu` resource requests) was scoped out of this submission.
