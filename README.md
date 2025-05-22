# churn-prediction
# Churn Prediction Pipeline

An end-to-end customer churn prediction system using the Telco Customer Churn dataset.  
This project covers data ingestion & cleaning, feature engineering, model training & comparison, evaluation, and deployment as a Flask API.

---

## 📁 Project Structure

```text
churn-prediction/
├── data/
│   └── Telco-Customer-Churn.csv      # Raw dataset
├── notebooks/
│   └── churn_eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py            # load_and_clean_data()
│   ├── feature_engineering.py        # engineer_features()
│   ├── model_training.py             # train_models(), save artifacts
│   ├── evaluate_model.py             # plot_roc()
│   └── predict_api.py                # Flask inference endpoint
├── tests/
│   └── test_api.py                   # API smoke test using `requests`
├── churn_model.pkl                   # Trained model artifact
├── scaler.pkl                        # StandardScaler for inference
├── columns.pkl                       # List of training columns
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # (Optional) Docker container spec
├── .gitignore
└── README.md                         # This file
```
## ⚙️ Setup & Installation

### Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction.git
cd churn-prediction
```

### Create & activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ Model Training

Run the full pipeline (data load → feature engineering → model comparison → artifact saving):

```bash
python -m src.model_training
```

Artifacts saved to project root:

* `churn_model.pkl`
* `scaler.pkl`
* `columns.pkl`

---

## 🚀 Running the Flask API

Start the inference server:

```bash
python -m src.predict_api
```

By default it listens on:
`http://127.0.0.1:5000`

### Health-check (optional)

If enabled, a GET request to `/` will return:
`API is live!`

---

## 🧪 Testing the API

Use the provided Python script or PowerShell commands.

### Python

```bash
python tests/test_api.py
# Should print: {'churn_probability': [0.xxx]}
```

### PowerShell

```powershell
$body = @'
[{
  "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
  "tenure":12,"PhoneService":"Yes","MultipleLines":"No",
  "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"Yes",
  "DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
  "StreamingMovies":"No","Contract":"Month-to-month",
  "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
  "MonthlyCharges":70.35,"TotalCharges":845.5
}]
'@

Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST `
  -ContentType 'application/json' -Body $body
```

---

## 🐳 Docker (Optional)

Build & run via Docker:

```bash
docker build -t churn-api .
docker run -p 5000:5000 churn-api
```

Your API will then be accessible at:
`http://localhost:5000`

---

## 📄 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and open a Pull Request

Please ensure your code follows **PEP8** and includes tests for any new functionality.

---

## 📜 License

This project is MIT-licensed. See [LICENSE](LICENSE) for details.
