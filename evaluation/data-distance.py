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
    data_path = "time_teacher_4000_z_dim_50_c_1e-4/samples/0.9/private.data"
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
    public_data = public_data[:data_len]
    
    mse = ((private_data - public_data)**2).mean(axis = 1)
    print(mse.shape)
    # mse = mse[outliers_iqr(mse)]
    sns.boxplot(mse)
    plt.savefig('result.png')

