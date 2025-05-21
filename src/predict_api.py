from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Make sure src package is discoverable
# If necessary, adjust PYTHONPATH or use absolute imports
from src.feature_engineering import engineer_features

app = Flask(__name__)

# Load artifacts
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
expected_cols = joblib.load('columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # 1) Read raw JSON into DataFrame
    data_raw = pd.DataFrame(request.json)

    # 2) Apply feature engineering
    data_fe = engineer_features(data_raw)

    # 3) Scale numeric features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data_fe[num_cols] = scaler.transform(data_fe[num_cols])

    # 4) Align columns to training schema
    data_aligned = data_fe.reindex(columns=expected_cols, fill_value=0)

    # 5) Predict probabilities
    preds = model.predict_proba(data_aligned)[:, 1]

    return jsonify({'churn_probability': preds.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
