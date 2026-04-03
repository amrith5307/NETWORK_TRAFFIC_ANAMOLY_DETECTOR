import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Near-constant columns in NSL-KDD — drop to reduce noise
COLS_TO_DROP = ['num_outbound_cmds', 'is_host_login']

def preprocess_data(df, is_train=True, encoder_dict=None):
    if df.shape[1] >= 42:
        df = df.iloc[:, :42]

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1]

    # FIX: always cast to string before comparison
    y = y.astype(str).str.strip().str.lower()
    y = (y != 'normal').astype(int)   # 0 = normal, 1 = attack

    # Drop noisy near-constant columns
    drop_cols = [c for c in COLS_TO_DROP if c in X.columns]
    X = X.drop(columns=drop_cols)

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    if is_train:
        encoder_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoder_dict[col] = le

        # StandardScaler - This is now our final step
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Store original feature names for SHAP later
        encoder_dict['feature_names'] = X.columns.tolist()
        encoder_dict['scaler'] = scaler

        return X_scaled, y, encoder_dict

    else:
        for col in categorical_cols:
            le = encoder_dict[col]
            X[col] = X[col].astype(str).map(
                lambda s: s if s in le.classes_ else le.classes_[0]
            )
            X[col] = le.transform(X[col])

        X_scaled = encoder_dict['scaler'].transform(X)

        return X_scaled, y