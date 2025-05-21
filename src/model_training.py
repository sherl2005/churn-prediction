from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib

def train_models(df):
    X = df.drop('Churn_Yes', axis=1)
    y = df['Churn_Yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, preds)
        print(f"{name} ROC-AUC: {score:.4f}")
        if score > best_score:
            best_model = model
            best_score = score

    joblib.dump(best_model, 'churn_model.pkl')
