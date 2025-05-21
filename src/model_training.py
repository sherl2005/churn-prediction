from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os
# relative imports from this package:
from src.data_processing      import load_and_clean_data
from src.feature_engineering  import engineer_features

def train_models(df):
    X = df.drop('Churn_Yes', axis=1)
    y = df['Churn_Yes']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )

    # --- scale numeric features ---
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    # --- define models with higher max_iter for LR ---
    models = {
        'Logistic Regression': LogisticRegression(max_iter=5000, solver='lbfgs'),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost': XGBClassifier(eval_metric='logloss')
    }

    best_model, best_score = None, 0.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, preds)
        print(f"{name} ROC-AUC: {score:.4f}")
        if score > best_score:
            best_model, best_score = model, score

    # Save artifacts
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(root_dir, 'churn_model.pkl')
    scaler_path = os.path.join(root_dir, 'scaler.pkl')
    cols_path  = os.path.join(root_dir, 'columns.pkl')

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    # save expected columns for API alignment
    expected_cols = X_train.columns.tolist()
    joblib.dump(expected_cols, cols_path)
    print(f"Saved best_model ({best_score:.4f}), scaler, and columns list to {root_dir}")

if __name__ == "__main__":
    # find the project root dynamically
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_fp = os.path.join(base_dir, 'data', 'Telco-Customer-Churn.csv')

    df = load_and_clean_data(data_fp)
    df = engineer_features(df)
    train_models(df)
