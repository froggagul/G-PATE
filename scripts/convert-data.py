import sys
sys.path.insert(0, '.')

import joblib
from data import LGITDataset
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q1 - q3
    lower_bound = q1 - iqr * 1.5
    upper_bound = q3 + iqr * 1.5

    return np.where((data>lower_bound)|(data<upper_bound))


if __name__ == "__main__":
    eps = "1.9"
    data_dir = f"time_teacher_4000_z_dim_50_c_1e-4/samples/{eps}/"

    data_path = os.path.join(data_dir, "private.data")
    output_csv_path = os.path.join(data_dir, f"time-eps{eps}.csv")

    private_data = joblib.load(data_path)
    print(private_data.shape)

    data_dir = "data/time"
    data_df = pd.read_csv(os.path.join(data_dir, 'time.csv'))
    columns_df = pd.read_csv(os.path.join(data_dir, 'columns.csv'))

    dataset = LGITDataset(data_df, columns_df, batch_first = True)
    public_data = dataset.input_data
    print(public_data.shape)

    data_len = min(private_data.shape[0], public_data.shape[0])

    private_data = private_data[:data_len]
    
    # convert
    
    private_columns = np.concatenate([dataset.T_columns, dataset.SA_columns, dataset.SB_columns])
    print(private_columns.shape)
    private_df = data_df.copy()[private_columns]

    private_df.loc[:, dataset.T_columns] = private_data[:, :len(dataset.T_columns)]
    private_df.loc[:, dataset.SA_columns] = private_data[:, len(dataset.T_columns):len(dataset.T_columns) + len(dataset.SA_columns)]
    private_df.loc[:, dataset.SB_columns] = private_data[:, len(dataset.T_columns) + len(dataset.SA_columns):len(dataset.T_columns) + len(dataset.SA_columns) + len(dataset.SB_columns)]

    private_df.to_csv(output_csv_path)

