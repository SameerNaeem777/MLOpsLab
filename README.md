# MLOps Lab

A reproducible customer churn classification demo with a FastAPI inference service, a Docker image, and Kubernetes manifests.

## Project layout

- app/main.py: HTTP health and prediction endpoints.
- scripts/generate_data.py: deterministic synthetic dataset generator.
- scripts/train_model.py: preprocessing and Random Forest training.
- k8s/: Kubernetes namespace, optional PostgreSQL, and API resources.
- archive/legacy/: original files retained for reference; these are not used by the new deployment.

## Run locally

Use Python 3.11 or newer. From the project root, run:

    python -m venv .venv
    & ./.venv/Scripts/Activate.ps1
    python -m pip install -r requirements.txt
    python scripts/generate_data.py
    python scripts/train_model.py
    uvicorn main:app --app-dir app --reload

The API documentation is available at http://127.0.0.1:8000/docs. Send a POST request to /predict with JSON such as:

    {
      "tenure_months": 5,
      "monthly_charges": 89.5,
      "total_charges": 447.5,
      "support_calls": 3,
      "contract_type": "Month-to-Month",
      "internet_service": "Fiber Optic"
    }

Omit total_charges or set it to null to have it calculated from tenure and monthly charges.

## Build and run with Docker

The image build generates the dataset and trains the model inside the image, so the serialized model uses the same scikit-learn version as the API runtime.

    docker build -t ml-model-api:v2 .
    docker run --rm -p 8000:8000 ml-model-api:v2

## Deploy the API to Kubernetes

This project's Kubernetes context is Minikube. Apply the namespace, build the image, load it into Minikube, and apply the API resources:

    kubectl apply -f k8s/namespace.yaml
    docker build -t ml-model-api:v2 .
    & "C:\Program Files\Kubernetes\Minikube\minikube.exe" image load ml-model-api:v2
    kubectl apply -f k8s/ml-api.yaml
    kubectl -n mlops-lab rollout status deployment/ml-inference-deployment
    kubectl -n mlops-lab port-forward service/ml-api-service 8000:8000

The Minikube executable can also be run as minikube if its folder is on PATH. For kind, use kind load docker-image ml-model-api:v2. For a remote cluster, push the image to a registry and update the image field in k8s/ml-api.yaml.

The API is then available at http://127.0.0.1:8000 and its interactive documentation is at http://127.0.0.1:8000/docs.

## Optional PostgreSQL deployment

The API does not currently store predictions in PostgreSQL. The database manifest is kept as an optional infrastructure exercise. Create the secret in the same namespace before applying the database manifest; replace the sample password with a private value:

    kubectl -n mlops-lab create secret generic postgres-secret --from-literal=POSTGRES_USER=ml_user --from-literal=POSTGRES_PASSWORD='choose-a-private-password' --from-literal=POSTGRES_DB=mldb
    kubectl apply -f k8s/postgres.yaml

Do not commit real database credentials. The PostgreSQL manifest creates its persistent volume claim. The cluster needs a default storage class or another provisioned volume.

## Notes

The dataset and model are generated locally and ignored by Git. The scripts use fixed random seeds for repeatable training data. The archive directory contains the original project copy and is not part of the active deployment.
