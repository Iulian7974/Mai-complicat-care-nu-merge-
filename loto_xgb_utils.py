import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter

def load_data(file):
    df = pd.read_excel(file)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.dropna(subset=['Data'])
    return df

def prepare_data(df):
    X, y = [], []
    for i in range(len(df) - 1):
        features = df.iloc[i][['Nr.1','Nr.2','Nr.3','Nr.4','Nr.5','Nr.6']].values
        targets = df.iloc[i+1][['Nr.1','Nr.2','Nr.3','Nr.4','Nr.5','Nr.6']].values
        for t in targets:
            X.append(features)
            y.append(t)
    return np.array(X), np.array(y)

def train_model(df):
    X, y = prepare_data(df)
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=50)
    model.fit(X, y)
    return model

def predict_next_draw(model, df):
    last_row = df.iloc[-1][['Nr.1','Nr.2','Nr.3','Nr.4','Nr.5','Nr.6']].values.reshape(1, -1)
    probs = model.predict_proba(last_row)
    avg_probs = np.mean(probs, axis=0)
    top_10 = np.argsort(avg_probs)[-10:][::-1]
    return top_10[:6], top_10
