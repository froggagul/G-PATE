import os

import torch
from torch.utils.data import Dataset
import pandas as pd

class LGITDataset(Dataset):
    def __init__(self, data_df, columns_df, batch_first=True) -> None:
        super().__init__()
        columns_df.index.name = 'index'
        SA_columns = columns_df[columns_df['type'] == 'SA'] # 58 kind, 116 columns
        self.SA_columns = SA_columns.sort_values(by=['sequence', 'index']).nor.values

        SB_columns = columns_df[columns_df['type'] == 'SB'] # 105 kind, 630 columns
        self.SB_columns = SB_columns.sort_values(by=['sequence', 'index']).nor.values

        T_columns = columns_df[columns_df['type'] == 'T']
        self.T_columns = T_columns.sort_index().nor.values

        OUT_columns = columns_df[columns_df['type'] == 'Y']
        self.OUT_columns = OUT_columns.sort_index().nor.values

        self.id = torch.tensor(data_df['ID'].values, dtype=torch.float32)
        self.party = torch.tensor(data_df['CAT_000'].values, dtype=torch.float32)
        self.tab_data = torch.tensor(data_df[self.T_columns].values, dtype=torch.float32)
        self.sa_data = torch.tensor(data_df[self.SA_columns].values, dtype=torch.float32)
        self.sb_data = torch.tensor(data_df[self.SB_columns].values, dtype=torch.float32)
        self.out = torch.tensor(data_df[self.OUT_columns].values, dtype=torch.float32)

        self.sa_len = 2
        self.sb_len = 6
        self.sa_dim = len(self.SA_columns) // self.sa_len
        self.sb_dim = len(self.SB_columns) // self.sb_len
        self.tab_dim = len(self.T_columns)
        self.out_dim = len(self.OUT_columns)

        self.input_data = torch.cat([self.tab_data, self.sa_data, self.sb_data], 1)
        self.input_data = self.input_data.numpy()

        if batch_first:
            # if batch_first = True(batch_size, seq_len, input_dim)
            self.sa_data = self.sa_data.reshape(-1, self.sa_len, self.sa_dim)
            self.sb_data = self.sb_data.reshape(-1, self.sb_len, self.sb_dim)
        else:
            # if batch_first = True(seq_len, batch_size, input_dim)
            self.sa_data = self.sa_data.reshape(self.sa_len, -1, self.sa_dim)
            self.sb_data = self.sb_data.reshape(self.sb_len, -1, self.sb_dim)
        
    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, i):
        return [self.tab_data[i], self.sa_data[i], self.sb_data[i]], self.out[i]

    def get_dims(self):
        return {'tab': self.tab_dim, 'sa': self.sa_dim, 'sb': self.sb_dim, 'out': self.out_dim}
