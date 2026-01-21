import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import Config, EarlyStopping
from model import CRNN
from dataset import TrainDataset, ACETestDataset

def main():
    print(f"Using device: {Config.DEVICE}")
    model = CRNN().to(Config.DEVICE)

    # ==========================================
    # A. 訓練流程
    # ==========================================
    if os.path.exists(Config.MODEL_PATH):
        print(f"\n[WARNING] 發現舊模型 {Config.MODEL_PATH}，如需重新訓練請先刪除它！")
        model.load_state_dict(torch.load(Config.MODEL_PATH)) 
    else:
        print("\n未找到模型，開始訓練流程...")
        print("\n準備訓練資料...")
        full_df = pd.read_csv(Config.TRAIN_CSV_PATH)
        
        # 篩選 T60 <= 1.3 秒的資料
        full_df = full_df[full_df['Measured_T60'] <= 1.3]
        
        TARGET_COUNT = 38782
        if len(full_df) > TARGET_COUNT:
            full_df = full_df.sample(n=TARGET_COUNT, random_state=42).reset_index(drop=True)

        train_df = full_df.iloc[:36693]
        val_df = full_df.iloc[36693:]

        train_ds = TrainDataset(train_df, cache_ram=True)
        val_ds = TrainDataset(val_df, cache_ram=True)

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=16)

        # 設定優化器與損失函數
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
        # 當 Val Loss 卡住 3 個 Epoch 不降，就將 LR 乘上 0.5
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler('cuda')
        # 初始化 Early Stopping
        early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=True, path=Config.MODEL_PATH)

        print("Start Training...")
        for epoch in range(Config.NUM_EPOCHS):
            model.train()
            t_loss = 0
            pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")

            for x, y in pbar:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE).unsqueeze(1)
                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    pred = model(x)
                    loss = criterion(pred, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Gradient Clipping 防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                t_loss += loss.item() * x.size(0)
                pbar.set_postfix({'loss': loss.item()})

            t_loss /= len(train_ds)

            # Validation
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE).unsqueeze(1)
                    p = model(x)
                    v_loss += criterion(p, y).item() * x.size(0)
            v_loss /= len(val_ds)

            print(f"Ep {epoch+1}: Train {t_loss:.4f}, Val {v_loss:.4f}")
            # 更新 Scheduler
            scheduler.step(v_loss)
            # 呼叫 Early Stopping
            early_stopping(v_loss, model)

            if early_stopping.early_stop:
                print("Early stopping triggered! 訓練提早結束。")
                break

    # ==========================================
    # B. 測試流程
    # ==========================================
    print("\nTesting ACE ...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH))
    
    model.eval()
    test_ds = ACETestDataset(Config.TEST_ROOT_DIR)
    
    if len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        preds, targets = [], []
        with torch.no_grad():
            for x, y in tqdm(test_loader):
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                p = model(x)
                preds.append(p.item())
                targets.append(y.item())
        
        preds, targets = np.array(preds), np.array(targets)
        
        mse = np.mean((preds - targets) ** 2)
        mae = np.mean(np.abs(preds - targets))
        bias = np.mean(preds - targets)
        pcc = np.corrcoef(preds, targets)[0, 1]
        
        print(f"\n===== 測試結果 =====")
        print(f"MSE  : {mse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"Bias : {bias:.4f}")
        print(f"PCC  : {pcc:.4f}")

        # 儲存結果與繪圖
        df = pd.DataFrame({
            "T60": targets,
            "Error": preds - targets
        })
        df["T60"] = df["T60"].round(1)
        df.to_csv("output.csv", index=False)
        
        flierprops = dict(marker='d', markerfacecolor='black', markersize=4, linestyle='none')
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="T60", y="Error", data=df, notch=False, palette="Set2", hue="T60", legend=False, width=0.6, flierprops=flierprops)
        plt.ylim(-1, 1)
        plt.xlabel("T60 [s]", fontsize=12)
        plt.ylabel("Error [s]", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig("ace_boxplot.png")
        print("結果圖已儲存為 ace_boxplot.png")

if __name__ == '__main__':
    main()