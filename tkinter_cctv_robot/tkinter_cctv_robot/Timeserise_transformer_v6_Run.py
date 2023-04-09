import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset
from torch.nn import Transformer
from torch import nn
import torch
import math
# from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#학습데이터 읽어 오기
rawdata = pd.read_csv('Shift.csv', encoding='CP949')
# plt.figure(figsize=(20,5))
# plt.plot(range(len(rawdata)), rawdata["Shift"])
rawdata.head()
#plt.show()

# -1 ~ 1 Normialize, 학습과 테스트 데이터 나누기
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
rawdata["Shift"] = min_max_scaler.fit_transform(rawdata["Shift"].to_numpy().reshape(-1,1))

# 학습데이터 70%, 테스트데이터 30%
trainsample = int(len(rawdata) * 0.7)

train = rawdata[:trainsample]
data_train = train["Shift"].to_numpy()

test = rawdata[trainsample:]
data_test = test["Shift"].to_numpy()


class windowDataset(Dataset):
    def __init__(self, y, input_window=80, output_window=20, stride=5):
        #총 데이터의 개수
        L = y.shape[0]
        #stride씩 움직일 때 생기는 총 sample의 개수
        num_samples = (L - input_window - output_window) // stride + 1

        #input과 output
        X = np.zeros([input_window, num_samples])
        Y = np.zeros([output_window, num_samples])

        for i in np.arange(num_samples):
            start_x = stride*i
            end_x = start_x + input_window
            X[:,i] = y[start_x:end_x]

            start_y = stride*i + input_window
            end_y = start_y + output_window
            Y[:,i] = y[start_y:end_y]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1,0,2))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1,0,2))
        self.x = X
        self.y = Y

        self.len = len(X)
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    def __len__(self):
        return self.len

class TFModel(nn.Module):
    def __init__(self,iw, ow, d_model, nhead, nlayers, dropout=0.5):
        super(TFModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
        )

        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask


# # model = torch.load('Transform_model.pt', map_location=torch.device('cpu'))
# model = torch.load('Transform_model.pt', map_location=device)
# # weights = torch.load('./Trained_Model/weights_gen.pt', device)
# model.eval()
#
# #inference
# # start = time.time()
# # test = np.random.rand(100)
#
#
# def transformerRun(valueArray):
#     print("zzz")
#
#     # input = torch.tensor(valueArray).reshape(1,-1,1).to(device).float().to(device)
#     # src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
#     # predictions = model(input, src_mask)
#     # result = predictions.detach().cpu().numpy()
#     # next = result[0][0]
#     # print(result)
#     # print("time: ", time.time() - start)
# #     return result

# model = TFModel(100, 10, 512, 8, 4, 0.1).to(device)
# model.load_state_dict(torch.load('Transform_model.pt'))

model = torch.load('Transform_model.pt')

# #inference
# model.eval()

#inference
# start = time.time()
# test = np.random.rand(100)


def transformerRun(valueArray):
    input = torch.tensor(valueArray).reshape(1,-1,1).to(device).float().to(device)
    src_mask = model.generate_square_subsequent_mask(input.shape[1]).to(device)
    predictions = model(input, src_mask)
    result = predictions.detach().cpu().numpy()
    next = result[0][0]
    # print(result)
    # print("time: ", time.time() - start)
    return result[0]