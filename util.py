import pandas as pd
import numpy as np

def prepare_dataset(data_file):
    data = pd.read_csv(data_file)
    int_columns = data.select_dtypes(include=['int64']).columns
    data[int_columns] = data[int_columns].astype(float)
    data.replace(-999999.0, 0.0, inplace=True)
    return data
