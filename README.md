# prodml

A minimal ML service that predicts NYC green taxi trip duration (in minutes) from pickup/dropoff location and trip distance, served via a FastAPI REST API.

## Quickstart

```bash
docker pull yassinaboelnaga/prodml-api:0.1.0
docker run -p 8000:8000 yassinaboelnaga/prodml-api:0.1.0
```

Then, in another terminal:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PU_DO": "74_42", "trip_distance": 3.2}'
```

## Repo structure
```
mlops-practitioner/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   ├── model.pkl
│   └── model.onnx
├── notebooks/
│   └── 00-baseline.ipynb
├── reports/
│   └── module-1.md
├── src/prodml/
│   ├── api/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── predict.py
│   └── train.py
├── tests/
├── pyproject.toml
└── README.md
```
