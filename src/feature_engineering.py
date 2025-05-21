import pandas as pd

def engineer_features(df):
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-12','13-24','25-48','49-72'])

    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['num_services'] = df[services].apply(lambda x: sum(x == 'Yes'), axis=1)

    df = pd.get_dummies(df, drop_first=True)
    return df
