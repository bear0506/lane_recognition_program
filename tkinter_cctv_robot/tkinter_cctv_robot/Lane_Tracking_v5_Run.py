import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

input_window = 10 # number of input steps
output_window = 1 # number of prediction steps, in this model its fixed to one
batch_size = 250
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

firstCheck = False
csum_logreturn = []

# df = pd.read_csv('test.csv') # 차선 도색 차량 추종 데이터
# close = np.array(df['Shift'])

# # Normalization with MinMax =====================================
minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
# log_normal = minmax_scaler.fit_transform(close.reshape(-1,1)).reshape(-1)
# logreturn = np.diff(log_normal) #훈련데이터
# #csum_logreturn = np.cumsum(np.append(log_normal[0], logreturn))
# csum_logreturn = log_normal

# Positional Encoder ========================================
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
# Transformer Model =======================================
class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
# ====================================================

# Window function, split data into sequence window ====================
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)
# ====================================================

# Split into training batches ====================================
def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target
# ====================================================

# Function to forecast 1 time step from window sequence =================
def model_forecast(model, seqence):
    model.eval() # 예측모드
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)

    seq = np.pad(seqence, (0, 3), mode='constant', constant_values=(0, 0))
    seq = create_inout_sequences(seq, input_window)
    seq = seq[:-output_window].to(device)

    seq, _ = get_batch(seq, 0, 1)
    with torch.no_grad():
        for i in range(0, output_window):            
            output = model(seq[-output_window:])                        
            seq = torch.cat((seq, output[-1:]))

    seq = seq.cpu().view(-1).numpy()

    return seq
# ====================================================

# Load model ============================================
model = TransAm().to(device)
model.load_state_dict(torch.load("transformer_ts.pth", map_location=device))
model.to(device)
# ====================================================

# 평가 모델 =============================================
def MSE(true, esti):
    return mean_squared_error(true, esti)

def MAE(true, esti):
    return mean_absolute_error(true, esti)
# ====================================================

def LaneTracking(shiftArray):
    global firstCheck
    global csum_logreturn
    # print(shiftArray)
    # Run 모듈 =============================================
    #일반 평가를 위해 기존 데이터에서 샘플 데이터를 추출합니다.
    #실행시에는 사용하지 않습니다.
    # r = np.random.randint(0, len(close)-1000)

    # (1) close[r:r+10] 대신에 10개의 Data를 선언하여 입력합니다. (np.array)
    # ex: close = np.array([x1, x2, ... x10])
    # (2) csum_logreturn으로 입력된 array를 MinMax scale로 변환합니다.
    # csum_logreturn = minmax_scaler.fit_transform(close[r:r+10].reshape(-1,1)).reshape(-1)
    # csum_logreturn = minmax_scaler.fit_transform(shiftArray.reshape(-1,1)).reshape(-1)

    # Minmax로 변환된 값을 모델을 통해 예측합니다. (10개를 넣으면, 11개를 반환)
    # 즉, 11번째 데이터가 예측값 입니다.
    # 10개의 값을 얻으로면, np.roll을 통해 기존값[9개] + 예측값[1개]를 다시 모델에 넣어 예측합니다.
    # test_forecast = model_forecast(model, csum_logreturn)

    # # 예측된 값을 다시 원래의 값으로 변환합니다.
    # result = minmax_scaler.inverse_transform(test_forecast.reshape(-1,1)).reshape(-1)
    # #[r: r+11]
    # print(f"Evaluation Index: {r}")
    # print("Evalution result ===========================")
    # print(f"Actual sequence: {csum_logreturn}")
    # print(f"Estimated sequence(difference): {test_forecast}")
    # print(f"Actual original input data: {close[r:r+11]}")
    # print(f"Estimated sequence(original data): {result}")
    # print(f"Accuracy (MSE): {MSE(close[r:r+11], result)}")
    # print(f"Accuracy (MAE): {MAE(close[r:r+11], result)}")
    # print("=====================================/n")
    # # ====================================================


    if firstCheck == False:
        csum_logreturn = minmax_scaler.fit_transform(shiftArray.reshape(-1, 1)).reshape(-1)

        for i in range(10):
            test_forecast = model_forecast(model, csum_logreturn)
            csum_logreturn = test_forecast[1:11]

        firstCheck = True
    else:
        test_forecast = model_forecast(model, csum_logreturn)
        csum_logreturn = test_forecast[1:11]


    result = minmax_scaler.inverse_transform(csum_logreturn.reshape(-1, 1)).reshape(-1)

    # print(result)
    # print(result[9])

    return result, result[9]

    # for i in range(10):
    #     print(csum_logreturn)
    #     test_forecast = model_forecast(model, csum_logreturn)
    #     csum_logreturn = test_forecast[1:11]
    #     print(test_forecast)
    #     print(i, "=========================")

    # result = minmax_scaler.inverse_transform(test_forecast[1:11].reshape(-1,1)).reshape(-1)
    # print("10 Step test ===========================")
    # print(f"Actual original data: {close[r+11:r+21]}")
    # print(f"Estimated sequence(original data): {result}")
    # print(f"Accuracy (MSE): {MSE(close[r+11:r+21], result)}")
    # print(f"Accuracy (MAE): {MAE(close[r+11:r+21], result)}")
    # print("=====================================")
