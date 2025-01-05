import torch
import torch.nn as nn
import pandas as pd
import argparse 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv
from torch.nn.functional import l1_loss
import torch.nn.functional as F
import os
import numpy as np

class Model(nn.Module):
    def __init__(self, win_size,input_dim,hidden_dim,t_att_heads,gru_layers,heads,out_dim,device):
        super(Model, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(win_size, t_att_heads)
        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
        self.gru2 = nn.GRU(input_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
        self.gru3 = nn.GRU(input_dim, hidden_dim, num_layers=gru_layers, batch_first=True)
        self.gat1 = GATConv(hidden_dim*win_size, hidden_dim, heads=heads) 
        self.gat2 = GATConv(hidden_dim*win_size, hidden_dim, heads=heads) 
        self.pn = PairNorm(mode='PN-SI')
        self.predictor = nn.Sequential(nn.Linear(in_features=hidden_dim*heads, out_features=out_dim,bias=True),
                                    nn.Sigmoid())
        self.alpha = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.decay = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.device= device
        
    def generate_edge_index(self, adj):
        I = torch.eye(adj.size(0)).to(self.device)
        adj+=I
        # 得到所有大于阈值的元素坐标
        indices = torch.nonzero(adj > adj.mean()+2*adj.std())
        # 转换为边索引
        edge_index = indices.t()
        return edge_index
    def forward(self, data, hs1):
        data = data.permute(0, 2, 1)
        hs1 = hs1.permute(0, 2, 1)
        embedding1, _ = self.multihead_attn(data, data, data)
        embedding1 = embedding1.permute(0, 2, 1)
        embedding2, _ = self.multihead_attn(hs1, hs1, hs1)
        embedding2 = embedding2.permute(0, 2, 1)
        out_GRU_1, _ = self.gru1(embedding1)
        out_GRU_2, _ = self.gru2(embedding1)
        out_GRU_3, _ = self.gru3(embedding2)
        
        M1 = torch.tanh(out_GRU_1.reshape(out_GRU_1.size(0), -1))
        M2 = torch.tanh(out_GRU_2.reshape(out_GRU_2.size(0), -1))
        M3 = torch.tanh(out_GRU_3.reshape(out_GRU_3.size(0), -1))
        
        adjacency_matrix_1 = F.relu(torch.matmul(M1, M2.t()) - torch.matmul(M2, M1.t()))
        adjacency_matrix_2 = F.relu(torch.matmul(M1, M3.t()) - torch.matmul(M3, M1.t()))
        edge_index_1 = self.generate_edge_index(adjacency_matrix_1)
        edge_index_2 = self.generate_edge_index(adjacency_matrix_2)
        GAT_output_1 = self.gat1((M1+M2)/2, edge_index_1)
        GAT_output_2 = self.gat2((M1+M3)/2, edge_index_2)
        embedding = self.pn(GAT_output_1*self.alpha+GAT_output_2*self.decay)
        output = self.predictor(embedding)
        return output


def filter_extreme_3sigma(series, n=3):
        mean = series.mean()
        std = series.std()
        max_range = mean + n * std
        min_range = mean - n * std
        return np.clip(series, min_range, max_range)

def standardize_zscore(series):
    std = series.std()
    mean = series.mean()
    return (series - mean) / std
    
def process_features(df_features, feature_cols):
    df_features_grouped = df_features.groupby('dt')
    res = []
    for dt in df_features_grouped.groups:
        df = df_features_grouped.get_group(dt)
        processed_df = process_daily_df_std(df, feature_cols)
        res.append(processed_df)
    df_features = pd.concat(res)
    df_features = df_features.dropna(subset=feature_cols)
    return df_features
def process_daily_df_std(df, feature_cols):
    df = df.copy()
    for c in feature_cols:
        df[c] = df[c].replace([np.inf, -np.inf], 0)
        df[c] = filter_extreme_3sigma(df[c])
        df[c] = standardize_zscore(df[c])
    return df

class StockDataset(Dataset):
    def __init__(self, standardize_df,win_size, horizon, mode, feature_cols, last_date):
        self.win_size = win_size
        self.horizon = horizon
        self.mode = mode
        
        self.dataset =  standardize_df
        self.dataset['dt'] = pd.to_datetime(self.dataset['dt']) 
        self.stocks = self.dataset['kdcode'].unique()
        self.feature_cols=feature_cols
        self.last_date=last_date
        # Preprocess data
        self.feature_slices, self.label_slices = self.preprocess_data()


    def preprocess_data(self):
        feature_slices = []
        label_slices = []
        
        for stock in self.stocks:
            stock_data = self.dataset[self.dataset['kdcode'] == stock] 
            train_data = stock_data[stock_data['dt'] < pd.to_datetime('2023-01-01')]
            test_data = stock_data[stock_data['dt']> pd.to_datetime(self.last_date)] #2022-11-18 for hs300
            if self.mode == 'train':
                data = train_data
            elif self.mode == 'test':
                data = test_data
            
            
            feature_data=data.loc[:, self.feature_cols]
            label_data = np.expand_dims(data['OT'].values, axis=1)
            feature_slices.append([feature_data[i:i+self.win_size] for i in range(len(data)-self.win_size)])
            label_slices.append([label_data[i+self.win_size] for i in range(len(data)-self.win_size)])
        self.date=data['dt']
        
        feature_slices = np.transpose(np.array(feature_slices),(1,0,2,3))
        label_slices = np.transpose(np.array(label_slices),(1,0,2))
        return feature_slices, label_slices
    
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.feature_slices[index]), torch.FloatTensor(self.label_slices[index])  
    def __len__(self):
        return len(self.feature_slices)
    

def collate_fn(batch):
    feature_slices = [item[0] for item in batch]
    label_slices = [item[1] for item in batch]
    feature_slices=torch.stack(feature_slices, dim=0).permute(1,0,2,3)
    label_slices=torch.stack(label_slices, dim=0).squeeze(-1).permute(1,0)
    return feature_slices.view(feature_slices.shape[0],-1,feature_slices.shape[3]),label_slices
class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x