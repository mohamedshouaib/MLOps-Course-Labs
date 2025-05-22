```markdown
# Churn Prediction API

This repository contains a Dockerized FastAPI application that serves a pre-trained machine learning model (Random Forest) for predicting customer churn. It includes testing, CI/CD deployment to AWS EC2, and monitoring with Prometheus and Grafana.

---

## 🗂️ Project Structure

```

.
├── app
│   ├── app.py                 # Main FastAPI app
│   ├── test.py                # Pytest script for unit/integration tests
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile             # Docker build file
│   └── models
│       └── random\_forest\_model.pkl  # Serialized ML model
├── monitoring
│   ├── dashboard.json         # Grafana dashboard config
│   ├── dashboard.yml          # Grafana dashboard provisioning
│   ├── datasource.yml         # Grafana datasource configuration
│   └── prometheus.yml         # Prometheus scrape config
├── docker-compose.yaml        # Multi-container app (API + monitoring)
├── .github/workflows/deploy.yaml  # GitHub Actions CI/CD pipeline
├── .gitignore
└── README.md

````

---

## 🚀 Setup & Installation

1. **Clone the repository**

```bash
git clone https://github.com/mohamedshouaib/MLOps-Course-Labs.git
cd churn-app
````

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r app/requirements.txt
```

---

## 🧠 Model Info

* **Model Type:** Random Forest Classifier
* **Location:** `app/models/random_forest_model.pkl`
* **Prediction Task:** Binary classification - Customer churn (1 = will churn, 0 = will stay)

---

## 🖥️ Run the API Locally

```bash
cd app
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Docs UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔍 API Endpoints

| Endpoint   | Method | Description                  |
| ---------- | ------ | ---------------------------- |
| `/`        | GET    | Home message                 |
| `/health`  | GET    | API health check             |
| `/predict` | POST   | Predict churn from JSON data |

📤 **Sample input to `/predict`**:

```json
{
  "CreditScore": 700,
  "Age": 35,
  "Tenure": 5,
  "Balance": 50000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 75000.0,
  "Gender": 1,
  "Geography_Germany": 0,
  "Geography_Spain": 1
}
```

---

## ✅ Run Tests

```bash
cd app
pytest test.py -v
```

---

## 🐳 Docker Usage

### 1. Build Image

```bash
docker build -t churn-api:latest ./app
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

This spins up:

* The FastAPI application
* Prometheus for metrics
* Grafana for dashboards

---

## 📊 Monitoring Stack

Configuration files in `monitoring/`:

* Prometheus is configured via `prometheus.yml`
* Grafana dashboards and datasources are provisioned on startup.

To access Grafana:

```
http://localhost:3000
Login: admin / admin
```

---

## 🚀 CI/CD (GitHub Actions → AWS EC2)

Located at `.github/workflows/deploy.yaml`:

* On push to `main`:

  * Runs `pytest`
  * Builds and pushes a Docker image to AWS ECR
  * SSH into EC2
  * Pulls the latest repo and runs updated containers

> 🔐 Make sure GitHub secrets are configured:
> `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `ECR_REPO`, `EC2_HOST`, `EC2_SSH_KEY`

---

## 📬 Contact

Feel free to open an issue or PR for improvements or questions.
