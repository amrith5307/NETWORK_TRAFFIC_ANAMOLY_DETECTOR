import pandas as pd

def load_data(file_path):
    print("Loading dataset...")

    df = pd.read_csv(file_path, sep='\t', header=None)

    print("Data loaded successfully.")

    return df