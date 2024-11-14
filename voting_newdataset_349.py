import pandas as pd
import numpy as np
import torch
import random
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim

# 시드 고정 설정
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# CUDA 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# LSTM Autoencoder 모델 정의
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout=0.3):
        super(LSTMAutoencoder, self).__init__()
        
        # 단방향 LSTM Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.hidden_to_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )
        
        # Latent to Hidden for Decoder
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 단방향 LSTM Decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size, sequence_length, _ = x.size()
        
        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.hidden_to_latent(hidden[-1])  # 마지막 LSTM 층의 hidden state 사용

        # Decode
        decoder_hidden = self.latent_to_hidden(latent).unsqueeze(0).expand(self.decoder_lstm.num_layers, batch_size, -1).contiguous()
        decoder_input = torch.zeros((batch_size, sequence_length, hidden_dim)).to(x.device)
        
        # 단방향 LSTM을 사용한 Decoder 실행
        decoder_output, _ = self.decoder_lstm(decoder_input, (decoder_hidden, torch.zeros_like(decoder_hidden)))
        
        # 최종 재구성 단계
        x_reconstructed = self.output_layer(decoder_output)
        
        return x_reconstructed


class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super(GRUAutoencoder, self).__init__()
        
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, input_dim)
        self.decoder_gru = nn.GRU(input_dim, input_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        batch_size, sequence_length, _ = x.size()
        _, hidden = self.encoder_gru(x)
        latent = self.hidden_to_latent(hidden[-1])
        decoder_hidden = self.latent_to_hidden(latent).unsqueeze(0).expand(self.decoder_gru.num_layers, batch_size, -1).contiguous()
        decoder_input = torch.zeros((batch_size, sequence_length, input_dim)).to(x.device)
        x_reconstructed, _ = self.decoder_gru(decoder_input, decoder_hidden)
        
        return x_reconstructed

# CNN Autoencoder 모델 정의
class CNNAutoencoder(nn.Module):
    def __init__(self, input_dim, sequence_length, latent_dim):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder: Conv1D layers with increased complexity
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(sequence_length * 256, latent_dim),
            nn.Dropout(0.4)  # Dropout을 조금 더 증가
        )
        
        # Decoder: ConvTranspose1D layers with mirrored structure
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, sequence_length * 256),
            nn.Unflatten(1, (256, sequence_length)),
            nn.ConvTranspose1d(256, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Tanh 활성화 함수로 변경하여 -1에서 1로 제한
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, sequence_length, input_dim) -> (batch, input_dim, sequence_length)
        latent = self.encoder(x)
        x_reconstructed = self.decoder(latent)
        x_reconstructed = x_reconstructed.permute(0, 2, 1)  # 원래 형식으로 변환
        return x_reconstructed

# 모델 설정
input_dim = 51
hidden_dim = 64
latent_dim = 16
num_layers = 2
sequence_length = 5
threshold = 0.025

# 각 모델 인스턴스 생성 및 모델 리스트
lstm_model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers)
gru_model = GRUAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers)
cnn_model = CNNAutoencoder(input_dim=input_dim, sequence_length=sequence_length, latent_dim=latent_dim)
models = [lstm_model.to(device), gru_model.to(device), cnn_model.to(device)]

# 데이터 로드 및 시퀀스 생성
def load_and_preprocess_data(data_path, sequence_length):
    data = pd.read_csv(data_path)
    data.interpolate(method='linear', limit_direction='both', inplace=True)
    data.fillna(0, inplace=True)
    
    scaler = MinMaxScaler()
    data_values = data.drop(columns=["Timestamp"])
    scaled_data = scaler.fit_transform(data_values)
    
    def create_sequences(data, sequence_length):
        return np.array([data[i:i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    
    return create_sequences(scaled_data, sequence_length)

# 모델 학습 함수
def train_model(model, train_data, epochs=30, learning_rate=0.001):
    if model == lstm_model:
        epochs = 40
    elif model == gru_model:
        epochs = 30
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_data)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

# 학습 및 이상 탐지
def train_and_detect_anomalies(models, data, threshold):
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    votes = np.zeros(len(data), dtype=int)
    reconstruction_errors = []
    model_anomaly_counts = {}
    
    # 각 모델을 학습 후 이상 탐지
    for model in models:
        model_name = model.__class__.__name__
        train_model(model, data_tensor)  # 모델 학습
        model.eval()
        
        with torch.no_grad():
            predictions = model(data_tensor)
            reconstruction_error = torch.mean((predictions - data_tensor)**2, dim=[1, 2])
            reconstruction_errors.append(reconstruction_error.cpu().numpy())
            
            # 이상치로 판별된 데이터 수 계산
            anomaly_count = np.sum(reconstruction_error.cpu().numpy() > threshold)
            model_anomaly_counts[model_name] = anomaly_count
            
            # 임계값 기준으로 정상 데이터에 투표
            votes += (reconstruction_error <= threshold).cpu().numpy().astype(int)
    
    # 모델별 이상치 수 출력
    for model_name, count in model_anomaly_counts.items():
        print(f"{model_name} 모델의 이상치 개수: {count}")
    
    # 보팅 기반 정상/이상 탐지
    majority_vote_threshold = len(models) // 2 + 1
    is_normal = votes >= majority_vote_threshold
    normal_data = data[is_normal]
    
    print(f"Normal data count (majority voting): {np.sum(is_normal)}")
    print(f"Anomaly data count (majority voting): {len(is_normal) - np.sum(is_normal)}")
    
    return normal_data

# 학습 및 이상 탐지
def train_and_detect_anomalies(models, data, threshold):
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
    votes = np.zeros(len(data), dtype=int)
    reconstruction_errors = []
    model_anomaly_counts = {}
    
    # 각 모델을 학습 후 이상 탐지
    for model in models:
        model_name = model.__class__.__name__
        train_model(model, data_tensor)  # 모델 학습
        model.eval()
        
        with torch.no_grad():
            predictions = model(data_tensor)
            reconstruction_error = torch.mean((predictions - data_tensor)**2, dim=[1, 2])
            reconstruction_errors.append(reconstruction_error.cpu().numpy())
            
            # 이상치로 판별된 데이터 수 계산
            anomaly_count = np.sum(reconstruction_error.cpu().numpy() > threshold)
            model_anomaly_counts[model_name] = anomaly_count
            
            # 임계값 기준으로 정상 데이터에 투표
            votes += (reconstruction_error <= threshold).cpu().numpy().astype(int)
    
    # 모델별 이상치 수 출력
    for model_name, count in model_anomaly_counts.items():
        print(f"{model_name} 모델의 이상치 개수: {count}")
    
    # 보팅 기반 정상/이상 탐지
    majority_vote_threshold = len(models) // 2 + 1
    is_normal = votes >= majority_vote_threshold
    
    print(f"Normal data count (majority voting): {np.sum(is_normal)}")
    print(f"Anomaly data count (majority voting): {len(is_normal) - np.sum(is_normal)}")
    
    return is_normal

# 이상 데이터 라벨링
def label_data_as_anomalies(timestamps, is_normal, sequence_length):
    # 시퀀스의 첫 번째 타임스탬프만 사용하여 라벨 생성
    anomaly_labels = np.where(is_normal, 0, 1).astype(int)
    
    # 라벨의 길이를 원래 데이터와 동일하게 맞추기 위해 padding 추가
    anomaly_labels_full = np.concatenate([np.zeros(sequence_length - 1), anomaly_labels]).astype(int)
    anomaly_labels_full = anomaly_labels_full[:len(timestamps)]  # 길이 맞추기
    
    return pd.DataFrame({
        "Timestamp": timestamps,
        "Anomaly": anomaly_labels_full
    })

# 정상 데이터만 추출
data_path = "C:/Users/이세연/Downloads/reconstructed_con_sig_2.csv"
data = load_and_preprocess_data(data_path, sequence_length)
is_normal = train_and_detect_anomalies(models, data, threshold)

# Timestamp 불러와서 이상치 1, 정상 0으로 라벨링
timestamps = pd.read_csv(data_path)["Timestamp"]
output_df = label_data_as_anomalies(timestamps, is_normal, sequence_length)
output_df.to_csv("labeled_output.csv", index=False)
print("Results saved to labeled_output.csv")
