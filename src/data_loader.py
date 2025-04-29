import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return {
        "max": df["max_value"].iloc[0]
    }
