import torch
import torch.nn as nn
import torchaudio
from nnAudio.features import Gammatonegram
from .utils import Config
# ==========================================
# Preprocessing (nnAudio Version)
# ==========================================
class Preprocessing(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. A-Weighting 
        self.register_buffer('b_aw', torch.tensor([1.0, -1.99004745483398, 0.990072250366211]))
        self.register_buffer('a_aw', torch.tensor([1.0, -1.97503185272217, 0.975056648254394]))
        
        # 2. Gammatonegram 
        self.gammatone = Gammatonegram(
            sr=Config.SAMPLE_RATE,
            n_fft=Config.NN_N_FFT,
            n_bins=Config.N_BANDS,
            hop_length=Config.NN_HOP_LEN,
            fmin=400,
            fmax=6000,
            window='hann',
            center=True,
            trainable_bins = False,
            trainable_STFT = False,
            verbose=False
        )
        
    def forward(self, waveform):
        # waveform: [Batch, 1, Time]
        with torch.cuda.amp.autocast(enabled=False):
            # 確保輸入是 float32
            waveform = waveform.float()
            # 1. A-Weighting
            # lfilter 需要 [Batch, Channel, Time]
            weighted = torchaudio.functional.lfilter(waveform, self.a_aw, self.b_aw)
        
            # 2. Gammatonegram 
            # nnAudio 預期輸入 [Batch, Time] (如果 channel=1) 
            # 輸出 [Batch, n_bins, Time_Frames]
            spec = self.gammatone(weighted.squeeze(1)) 

            # 輸出的 spec 已經是 Power (Energy) 譜了，不需要再平方
            # 3. Log + Median Norm + Standardization 
            log_spec = torch.log(spec + 1e-6)

            # Median Subtraction
            median_val = torch.median(log_spec, dim=2, keepdim=True)[0]
            centered = log_spec - median_val

            # Standardization
            mean = centered.mean(dim=(1, 2), keepdim=True)
            std = centered.std(dim=(1, 2), keepdim=True)

            final_feature = (centered - mean) / (std + 1e-6)

        return final_feature # [Batch, 21, 2001] (長度取決於 n_fft/hop)

# ==========================================
# Model Architecture 
# ==========================================
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.preprocess = Preprocessing()
        
        # 6 Convolutional Layers
        # 透過 MaxPool 逐步將 Freq 維度 (21) 壓縮至 1
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Freq: 21 -> 10
            
            # Conv 2
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Freq: 10 -> 5
            
            # Conv 3
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Freq: 5 -> 2
            
            # Conv 4
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # No pool here
            
            # Conv 5
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Freq: 2 -> 1
            
            # Conv 6
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Freq is 1
        )
        
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # x: [Batch, 1, Time]
        x = self.preprocess(x) # [Batch, 21, T]
        x = x.unsqueeze(1)     # [Batch, 1, 21, T]
        x = self.features(x)   # [Batch, 256, 1, T]
        
        # [B, C, 1, T] -> [B, T, C]
        x = x.squeeze(2).permute(0, 2, 1) 
        
        out, _ = self.lstm(x)
        # x = out[:, -1, :] # Last time step
        x = torch.mean(out, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
