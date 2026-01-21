import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import soundfile as sf
from tqdm import tqdm
from .utils import Config
# ==========================================
# Dataset
# ==========================================
class TrainDataset(Dataset):
    def __init__(self, df, cache_ram=True):
        self.cache_ram = cache_ram
        self.data_cache = []
        
        
        print(f"正在篩選訓練資料 (保留 >= 4秒)... 原始數量: {len(df)}")
        valid_indices = []
        skipped_count = 0
        
        # 1. 篩選長度
        for idx in tqdm(range(len(df)), desc="Filtering"):
            try:
                row = df.iloc[idx]
                file_path = row['File']
                
                # 使用 sf.info 快速檢查，不讀取整個音檔
                info = sf.info(file_path)
                if info.duration >= 4.0:
                    valid_indices.append(idx)
                else:
                    skipped_count += 1
            except:
                skipped_count += 1
                
        # 更新 self.df，只留下合法的 row
        self.df = df.iloc[valid_indices].reset_index(drop=True)
        print(f"篩選完成！剩餘 {len(self.df)} 筆 (剔除 {skipped_count} 筆過短資料)")

        # 2. 載入 RAM (可選)
        if self.cache_ram:
            print(f"預載入 {len(self.df)} 筆資料到 RAM (Float16)...")
            for idx in tqdm(range(len(self.df)), desc="Caching"):
                try:
                    row = self.df.iloc[idx]
                    wav = self._load_wav(row['File'])
                    lbl = row['Measured_T60']
                    self.data_cache.append((wav.half(), lbl)) 
                except:
                    # 萬一讀取真的失敗，補全 0 防止崩潰
                    self.data_cache.append((torch.zeros(1, Config.N_SAMPLES).half(), 0.0))
            print("RAM 載入完成")

    def _load_wav(self, path):
        wav_numpy, sr = sf.read(path)
        wav = torch.from_numpy(wav_numpy).float()
        if wav.ndim == 1: wav = wav.unsqueeze(0)
        else: wav = wav.transpose(0, 1)
        # Mono
        if wav.shape[0] > 1: 
            wav = wav[0:1, :] 
        # Resample
        if sr != Config.SAMPLE_RATE: 
            wav = T.Resample(sr, Config.SAMPLE_RATE)(wav)

        if wav.shape[1] > Config.N_SAMPLES: 
            wav = wav[:, :Config.N_SAMPLES]
        elif wav.shape[1] < Config.N_SAMPLES: 
            # wav = torch.nn.functional.pad(wav, (0, N_SAMPLES - wav.shape[1]))
            n_repeat = int(np.ceil(Config.N_SAMPLES / wav.shape[1]))
            wav = wav.repeat(1, n_repeat)
            # 截斷到64000 samples
            wav = wav[:, :Config.N_SAMPLES]
        return wav

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        if self.cache_ram:
            w, l = self.data_cache[idx]
            return w.float(), torch.tensor(l, dtype=torch.float32)
        else:
            row = self.df.iloc[idx]
            return self._load_wav(row['File']), torch.tensor(row['Measured_T60'], dtype=torch.float32)

class ACETestDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        
        # 1. 建立音檔索引 (File Map)
        print(f"正在建立音檔索引 (遍歷 {root_dir})...")
        self.file_map = {}
        # 搜尋所有 wav 檔案
        audio_files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
        for f in audio_files:
            fname = os.path.basename(f)
            self.file_map[fname] = f
            
        print(f"硬碟中實際找到 {len(self.file_map)} 個 wav 檔")

        
        # 2. 讀取 CSV 並匹配
        csv_files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
        print(f"找到 {len(csv_files)} 個測試 CSV")

        skipped_count = 0

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                
                # 去除欄位名稱中的冒號 (:) 和空格
                df.columns = [c.replace(':', '').strip() for c in df.columns]
                
                for _, row in df.iterrows():
                    try:
                        # 3. 檔名組合邏輯
                        # 處理 SNR: 強制轉 int (去除小數點) 例如 -1.0 -> -1
                        snr_raw = row.get('SNR', '')
                        try:
                            snr_val = int(float(snr_raw))
                        except:
                            snr_val = str(snr_raw).strip()

                        # 組合各個部分，並將 "空格" 替換為 "_"
                        parts = [
                            str(row.get('Mic config','')).strip().replace(" ", "_"),
                            str(row.get('Room','')).strip().replace(" ", "_"),
                            str(row.get('Room config','')).strip().replace(" ", "_"),
                            str(row.get('Talker','')).strip().replace(" ", "_"),
                            str(row.get('Utterance','')).strip().replace(" ", "_"),
                            str(row.get('Noise','')).strip().replace(" ", "_"),
                            f"{snr_val}dB"
                        ]
                        
                        fname = "_".join(parts) + ".wav"
                        
                        # 4. 匹配檢查
                        if fname in self.file_map:
                            wav_path = self.file_map[fname]

                            # 檢查音檔長度，只保留 >= 4秒 的檔案 
                            # 使用 sf.info 快速讀取 Header 資訊，不需要讀取整個檔案
                            info = sf.info(wav_path)
                            if info.duration < 4.0:
                                skipped_count += 1
                                continue # 跳過這筆資料

                            # 優先讀取 T60 GT FB
                            if 'T60 GT FB' in row:
                                label = float(row['T60 GT FB'])
                            elif 'T60' in row:
                                label = float(row['T60'])
                            else:
                                continue 
                                
                            self.data.append((wav_path, label))
                    except Exception: continue
            except Exception: continue
        
        print(f"篩選後 (Duration >= 4s) 剩餘 {len(self.data)} 筆資料 (剔除 {skipped_count} 筆過短資料)")

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        path, label = self.data[idx]
        
        # 改用 soundfile 直接讀取，繞過 torchaudio.load 的後端問題
    
        try:
            # sf.read 回傳 (samples, channels)
            wav_numpy, sr = sf.read(path)
            
            # 轉為 Tensor: float32
            wav = torch.from_numpy(wav_numpy).float()
            
            # 處理維度: soundfile 讀出來如果是 (Time,) 需轉為 (1, Time)
            # 如果是 (Time, Ch) 需轉為 (Ch, Time) 以符合 PyTorch 習慣
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.transpose(0, 1)
                
        except Exception as e:
            print(f"[ERROR] 無法讀取檔案: {path}")
            print(f"原因: {e}")
            # 回傳空 Tensor 防止崩潰 
            wav = torch.zeros(1, Config.N_SAMPLES)
            sr = Config.SAMPLE_RATE

        # 確保單聲道
        if wav.shape[0] > 1: 
            wav = wav[0:1, :]
        
        # 重採樣
        if sr != Config.SAMPLE_RATE: 
            resampler = T.Resample(orig_freq=sr, new_freq=Config.SAMPLE_RATE)
            wav = resampler(wav)
        
        # 固定長度
        if wav.shape[1] > Config.N_SAMPLES: 
            wav = wav[:, :Config.N_SAMPLES]
        elif wav.shape[1] < Config.N_SAMPLES: 
            # wav = torch.nn.functional.pad(wav, (0, Config.N_SAMPLES - wav.shape[1]))
            n_repeat = int(np.ceil(Config.N_SAMPLES / wav.shape[1]))
            wav = wav.repeat(1, n_repeat)
            # 再次截斷到剛好 N_SAMPLES
            wav = wav[:, :Config.N_SAMPLES]
            
        return wav, torch.tensor(label, dtype=torch.float32)