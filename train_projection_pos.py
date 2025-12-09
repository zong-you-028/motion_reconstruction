"""
訓練可學習投影矩陣 POS 模型

支援三種模型：
1. ProjectionMatrixPredictor - 基礎模型
2. ConstrainedProjectionPredictor - 帶約束（residual 模式）
3. TemporalProjectionPredictor - 時序模型（LSTM）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

from learnable_projection_pos import (
    ProjectionMatrixPredictor,
    ConstrainedProjectionPredictor,
    TemporalProjectionPredictor,
    LearnableProjectionPOS,
    ProjectionMatrixLoss
)


class RPPGDatasetForProjection(Dataset):
    """
    用於投影矩陣學習的數據集
    """
    def __init__(self, rgb_traces, ppg_signals, window_length=150, stride=15, mode='feature'):
        """
        Args:
            rgb_traces: list of (r, g, b) tuples
            ppg_signals: list of PPG arrays
            window_length: 窗口長度
            stride: 滑動步長
            mode: 'feature' 或 'sequence'
        """
        self.window_length = window_length
        self.mode = mode
        self.samples = []
        
        pos_processor = LearnableProjectionPOS(window_length=window_length, fs=30)
        
        for rgb, ppg in zip(rgb_traces, ppg_signals):
            r_trace, g_trace, b_trace = rgb
            
            min_len = min(len(r_trace), len(g_trace), len(b_trace), len(ppg))
            r_trace = r_trace[:min_len]
            g_trace = g_trace[:min_len]
            b_trace = b_trace[:min_len]
            ppg = ppg[:min_len]
            
            # 創建滑動窗口
            for start_idx in range(0, len(r_trace) - window_length + 1, stride):
                end_idx = start_idx + window_length
                
                sample = {
                    'r': r_trace[start_idx:end_idx],
                    'g': g_trace[start_idx:end_idx],
                    'b': b_trace[start_idx:end_idx],
                    'ppg': ppg[start_idx:end_idx]
                }
                
                # 提取特徵（用於feature模式）
                if mode == 'feature':
                    features = pos_processor.extract_window_features(
                        sample['r'], sample['g'], sample['b']
                    )
                    sample['features'] = features
                
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        result = {
            'ppg': torch.FloatTensor(sample['ppg'])
        }
        
        if self.mode == 'feature':
            result['features'] = torch.FloatTensor(sample['features'])
        else:
            # 序列模式
            rgb_seq = np.vstack([sample['r'], sample['g'], sample['b']])
            result['rgb_seq'] = torch.FloatTensor(rgb_seq)
        
        # 同時返回 RGB 用於計算 rPPG
        result['r'] = torch.FloatTensor(sample['r'])
        result['g'] = torch.FloatTensor(sample['g'])
        result['b'] = torch.FloatTensor(sample['b'])
        
        return result


def compute_rppg_with_learned_P(r_buf, g_buf, b_buf, P_matrix):
    """
    使用學習到的投影矩陣計算 rPPG
    
    Args:
        r_buf, g_buf, b_buf: (batch, window_length) 或 1D arrays
        P_matrix: (batch, 2, 3) 或 (2, 3)
    
    Returns:
        rppg: (batch, window_length) 或 1D array
    """
    # 確保是 torch tensor
    if not isinstance(r_buf, torch.Tensor):
        r_buf = torch.FloatTensor(r_buf)
        g_buf = torch.FloatTensor(g_buf)
        b_buf = torch.FloatTensor(b_buf)
        P_matrix = torch.FloatTensor(P_matrix)
    
    # 標準化
    mu_r = torch.mean(r_buf, dim=-1, keepdim=True)
    mu_g = torch.mean(g_buf, dim=-1, keepdim=True)
    mu_b = torch.mean(b_buf, dim=-1, keepdim=True)
    
    r_norm = r_buf / (mu_r + 1e-8) - 1
    g_norm = g_buf / (mu_g + 1e-8) - 1
    b_norm = b_buf / (mu_b + 1e-8) - 1
    
    # 堆疊 RGB: (batch, 3, window_length)
    if len(r_norm.shape) == 1:
        rgb_stack = torch.stack([r_norm, g_norm, b_norm])  # (3, window_length)
    else:
        rgb_stack = torch.stack([r_norm, g_norm, b_norm], dim=1)  # (batch, 3, window_length)
    
    # 投影: P @ RGB
    if len(P_matrix.shape) == 2:
        # 單一矩陣
        S = torch.matmul(P_matrix, rgb_stack)  # (2, window_length)
    else:
        # 批次矩陣
        S = torch.matmul(P_matrix, rgb_stack)  # (batch, 2, window_length)
    
    # POS 組合（使用標準 alpha）
    if len(S.shape) == 2:
        alpha = torch.std(S[0, :]) / (torch.std(S[1, :]) + 1e-8)
        rppg = S[0, :] + alpha * S[1, :]
    else:
        alpha = torch.std(S[:, 0, :], dim=-1, keepdim=True) / \
                (torch.std(S[:, 1, :], dim=-1, keepdim=True) + 1e-8)
        rppg = S[:, 0, :] + alpha * S[:, 1, :]
    
    # 去均值
    rppg = rppg - torch.mean(rppg, dim=-1, keepdim=True)
    
    return rppg


def train_projection_model(model, train_loader, val_loader, num_epochs=50,
                           learning_rate=0.001, device='cuda', save_dir='./projection_models',
                           model_type='feature'):
    """
    訓練投影矩陣模型
    
    Args:
        model: 投影矩陣預測模型
        train_loader, val_loader: 數據載入器
        num_epochs: 訓練輪數
        learning_rate: 學習率
        device: 'cuda' 或 'cpu'
        save_dir: 模型保存目錄
        model_type: 'feature' 或 'sequence'
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 損失函數（根據物理導向的設計）
    criterion = ProjectionMatrixLoss(
        lambda_time=1.0,      # 時域損失（主導）
        lambda_freq=0.5,      # 頻域損失（生理約束）
        lambda_ortho=0.05,    # 正交約束（輕微，維持結構）
        fs=30                 # 採樣率
    )
    
    # 標準 POS 矩陣（保留以備參考，但新損失函數不需要）
    # P_standard = torch.tensor([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]]).to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # ===== 訓練 =====
        model.train()
        train_loss = 0.0
        train_metrics = {'time': 0, 'freq': 0, 'ortho': 0}
        
        for batch_idx, batch in enumerate(train_loader):
            # 移到設備
            ppg_gt = batch['ppg'].to(device)
            r = batch['r'].to(device)
            g = batch['g'].to(device)
            b = batch['b'].to(device)
            
            optimizer.zero_grad()
            
            # 預測投影矩陣
            if model_type == 'feature':
                features = batch['features'].to(device)
                P_pred = model(features)  # (batch, 2, 3)
            else:
                rgb_seq = batch['rgb_seq'].to(device)
                P_pred = model(rgb_seq)  # (batch, 2, 3)
            
            # 使用預測的 P 計算 rPPG
            rppg_pred = compute_rppg_with_learned_P(r, g, b, P_pred)
            
            # 計算損失（新設計：時域 + 頻域 + 正交）
            loss, metrics = criterion(
                rppg_pred, ppg_gt, P_pred
            )
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            for key in ['time', 'freq', 'ortho']:
                if key in metrics:
                    train_metrics[key] += metrics[key]
        
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # ===== 驗證 =====
        model.eval()
        val_loss = 0.0
        val_correlations = []
        
        with torch.no_grad():
            for batch in val_loader:
                ppg_gt = batch['ppg'].to(device)
                r = batch['r'].to(device)
                g = batch['g'].to(device)
                b = batch['b'].to(device)
                
                # 預測
                if model_type == 'feature':
                    features = batch['features'].to(device)
                    P_pred = model(features)
                else:
                    rgb_seq = batch['rgb_seq'].to(device)
                    P_pred = model(rgb_seq)
                
                # 計算 rPPG
                rppg_pred = compute_rppg_with_learned_P(r, g, b, P_pred)
                
                # 損失
                loss, _ = criterion(rppg_pred, ppg_gt, P_pred)
                val_loss += loss.item()
                
                # 相關係數
                from scipy.stats import pearsonr
                for i in range(len(rppg_pred)):
                    corr, _ = pearsonr(
                        rppg_pred[i].cpu().numpy(),
                        ppg_gt[i].cpu().numpy()
                    )
                    val_correlations.append(corr)
        
        val_loss /= len(val_loader)
        mean_corr = np.mean(val_correlations)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 打印進度
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    Time: {train_metrics['time']:.4f}, "
              f"Freq: {train_metrics['freq']:.4f}, "
              f"Ortho: {train_metrics['ortho']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Correlation: {mean_corr:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_correlation': mean_corr,
            }, os.path.join(save_dir, 'best_projection_model.pth'))
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
    
    # 繪製訓練曲線
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'projection_training_curves.png'))
    plt.close()
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    print("訓練可學習投影矩陣 POS")
    print("=" * 60)
    
    # 設置設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}\n")
    
    # 生成測試數據
    print("生成測試數據...")
    from demo_usage import generate_synthetic_rppg_data
    
    rgb_traces = []
    ppg_signals = []
    
    for i in range(20):
        r, g, b, ppg, _ = generate_synthetic_rppg_data(
            duration=30, fs=30, hr=np.random.uniform(60, 100),
            motion_strength=np.random.uniform(0.2, 0.6)
        )
        rgb_traces.append((r, g, b))
        ppg_signals.append(ppg)
    
    # 分割數據
    split_idx = int(0.8 * len(rgb_traces))
    train_rgb = rgb_traces[:split_idx]
    train_ppg = ppg_signals[:split_idx]
    val_rgb = rgb_traces[split_idx:]
    val_ppg = ppg_signals[split_idx:]
    
    print(f"訓練集: {len(train_rgb)} 個受試者")
    print(f"驗證集: {len(val_rgb)} 個受試者\n")
    
    # 選擇模型類型
    print("選擇模型類型:")
    print("1. ProjectionMatrixPredictor (基礎，基於特徵)")
    print("2. ConstrainedProjectionPredictor (帶約束，residual)")
    print("3. TemporalProjectionPredictor (時序，LSTM)")
    
    model_choice = 2  # 預設使用約束模型
    
    if model_choice == 1:
        print("\n使用 ProjectionMatrixPredictor")
        model = ProjectionMatrixPredictor(input_dim=10, hidden_dim=64)
        model_type = 'feature'
        dataset_mode = 'feature'
    elif model_choice == 2:
        print("\n使用 ConstrainedProjectionPredictor")
        model = ConstrainedProjectionPredictor(input_dim=10, hidden_dim=64, use_residual=True)
        model_type = 'feature'
        dataset_mode = 'feature'
    else:
        print("\n使用 TemporalProjectionPredictor")
        model = TemporalProjectionPredictor(window_size=128, hidden_dim=64)
        model_type = 'sequence'
        dataset_mode = 'sequence'
    
    print(f"模型參數量: {sum(p.numel() for p in model.parameters())}\n")
    
    # 創建數據集
    print("創建數據集...")
    train_dataset = RPPGDatasetForProjection(
        train_rgb, train_ppg, window_length=150, stride=15, mode=dataset_mode
    )
    val_dataset = RPPGDatasetForProjection(
        val_rgb, val_ppg, window_length=150, stride=15, mode=dataset_mode
    )
    
    print(f"訓練樣本: {len(train_dataset)}")
    print(f"驗證樣本: {len(val_dataset)}\n")
    
    # 創建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 訓練
    print("開始訓練...")
    trained_model, train_losses, val_losses = train_projection_model(
        model, train_loader, val_loader,
        num_epochs=30,
        learning_rate=0.001,
        device=device,
        save_dir='./projection_models',
        model_type=model_type
    )
    
    print("\n訓練完成！")
    print(f"最佳驗證損失: {min(val_losses):.4f}")
    print(f"模型保存在: ./projection_models/best_projection_model.pth")