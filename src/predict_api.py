from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('churn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.DataFrame(request.json)
    preds = model.predict_proba(data)[:, 1]
    return jsonify({'churn_probability': preds.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
