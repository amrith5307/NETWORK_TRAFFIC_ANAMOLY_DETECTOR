import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder

def preprocess_data(df, is_train=True, encoder_dict=None):
    # 1. NSL-KDD usually has 43 columns. The 42nd is the label, 43rd is 'difficulty'
    # We only want the first 41 features and the 42nd label.
    if df.shape[1] >= 42:
        # Keep only first 42 columns (41 features + 1 label)
        df = df.iloc[:, :42]
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Convert labels to binary (Normal=0, Attack=1)
    y = y.apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)
    
    # 2. Fix Categorical Columns (Protocol, Service, Flag)
    # We use a dictionary to keep encoders consistent between Train and Test
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if is_train:
        encoder_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoder_dict[col] = le
        return X, y, encoder_dict
    else:
        for col in categorical_cols:
            le = encoder_dict[col]
            # Handle new categories in test set that weren't in train
            X[col] = X[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            X[col] = le.transform(X[col].astype(str))
        
        # FINAL SAFETY CHECK: Ensure X has exactly 41 features
        if X.shape[1] != 41:
            # If there's an extra column somehow, drop it
            X = X.iloc[:, :41]
            
        return X, y