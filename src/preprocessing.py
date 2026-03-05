import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA

def preprocess_data(df, is_train=True, encoder_dict=None):
    if df.shape[1] >= 42:
        df = df.iloc[:, :42]
    
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1]
    y = y.apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    if is_train:
        encoder_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoder_dict[col] = le
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Increased to 15 for better accuracy
        pca = PCA(n_components=15)
        X_pca = pca.fit_transform(X_scaled)
        
        encoder_dict['scaler'] = scaler
        encoder_dict['pca'] = pca
        return X_pca, y, encoder_dict
    else:
        for col in categorical_cols:
            le = encoder_dict[col]
            X[col] = X[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col].astype(str))
        
        X_scaled = encoder_dict['scaler'].transform(X)
        X_pca = encoder_dict['pca'].transform(X_scaled)
        return X_pca, y