# 📈 Sales Prediction API

A Machine Learning API to predict sales based on advertising budgets (TV, Radio, Newspaper).
This project features a **custom Linear Regression algorithm built from scratch using Numpy** (no Scikit-learn models), served via **FastAPI**, and containerized with **Docker**.

## 🛠 Tech Stack
- **Core:** Python, Numpy, Pandas
- **API:** FastAPI, Uvicorn, Pydantic
- **DevOps:** Docker

## 📂 Project Structure
```text
├── data/              # Raw and processed data
├── models/            # Trained artifacts (.npy files)
├── notebook/          # EDA and experiments
├── scripts/           # Training and processing pipelines
├── src/               # API source code (main.py, schemas.py)
├── Dockerfile
└── requirements.txt

```
## 🧠 Technical Details
### 1. Algorithm (src/model.py)

This project implements a CustomLinearRegression class:

    Prediction: y=X⋅w+bias

    Optimization: Uses Gradient Descent to iteratively update weights.

    Loss Function: Mean Squared Error (MSE).

### 2. Data Processing

    Scaling: Applies Standardization (Z-score normalization) to scale features to a standard distribution (Mean=0, Std=1) before training. This ensures the Gradient Descent algorithm converges faster and more accurately.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Build image
docker build -t sales-prediction .

# 2. Run container
docker run -d -p 8000:8000 --name sales-container sales-prediction

```

👉 **Access Swagger UI:** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)

### Option 2: Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Data processing
python -m scripts.data_processing

# 3. Train model (Generates weights in /models)
python -m scripts.train_pipeline

# 4. Start API Server
uvicorn src.main:app --reload

```

## 🔌 API Usage

**Endpoint:** `POST /predict`

**Request Body:**

```json
{
  "tv": 150.5,
  "radio": 25.0,
  "newspaper": 10.0
}

```

**Response:**

```json
{
  "sales_prediction": 15.45
}

```

## 👤 Author
Huy Quach

Github: [@QuachGHuy](https://github.com/QuachGHuy)

LinkedIn: [Gia Huy Quach](www.linkedin.com/in/gia-huy-quach)



