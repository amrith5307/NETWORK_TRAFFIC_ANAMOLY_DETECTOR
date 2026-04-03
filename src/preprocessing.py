import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Near-constant columns in NSL-KDD — drop to reduce noise
COLS_TO_DROP = ['num_outbound_cmds', 'is_host_login']

def preprocess_data(df, is_train=True, encoder_dict=None):
    if df.shape[1] >= 42:
        df = df.iloc[:, :42]

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1]

    # FIX: always cast to string before comparison — prevents .str accessor error
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

        # StandardScaler (better than RobustScaler for EllipticEnvelope)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 20 PCA components retains ~95%+ variance on NSL-KDD
        pca = PCA(n_components=20, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        explained = np.sum(pca.explained_variance_ratio_)
        print(f"PCA: 20 components retain {explained:.2%} of variance")

        encoder_dict['scaler'] = scaler
        encoder_dict['pca'] = pca

        return X_pca, y, encoder_dict

    else:
        for col in categorical_cols:
            le = encoder_dict[col]
            X[col] = X[col].astype(str).map(
                lambda s: s if s in le.classes_ else le.classes_[0]
            )
            X[col] = le.transform(X[col])

        X_scaled = encoder_dict['scaler'].transform(X)
        X_pca = encoder_dict['pca'].transform(X_scaled)

        return X_pca, y