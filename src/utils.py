import numpy as np
import torch
import os
# ==========================================
# 設定 (Configuration)
# ==========================================
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 論文參數
    SAMPLE_RATE = 16000
    DURATION = 4
    N_SAMPLES = SAMPLE_RATE * DURATION
    N_BANDS = 21 

    # nnAudio 設定 
    NN_N_FFT = 1024
    NN_HOP_LEN = 32

    BATCH_SIZE = 256
    NUM_EPOCHS = 100 
    LEARNING_RATE = 1e-4
    PATIENCE = 10 # Early Stopping 等待次數

    # 路徑 (請依實際情況修改)
    TRAIN_CSV_PATH = "/home/herby/Synthetic/rir_metadata.csv"
    TEST_ROOT_DIR = "/home/herby/ACE_Eval_dataset/Eval/Speech"
    MODEL_PATH = "best_crnn.pth"

# ==========================================
# Early Stopping
# ==========================================
class EarlyStopping:
    # 當 Validation Loss 停止下降時，提早結束訓練
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # 驗證集 Loss 下降時儲存模型
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss