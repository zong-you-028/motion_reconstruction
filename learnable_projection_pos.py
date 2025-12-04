"""
Learnable Projection Matrix POS (LP-POS)

核心創新：訓練整個投影矩陣 P(t)，而不只是 alpha 參數

標準 POS:
    P = [[0, 1, -1],      (固定)
         [-2, 1, 1]]
    
可學習投影 POS (LP-POS):
    P(t) = NeuralNet(features_t)  (動態預測 2x3 矩陣)
    
優勢：
    - 更大的自由度
    - 能學習最優的色彩空間投影
    - 對不同膚色、光照更魯棒
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import signal
from scipy.stats import pearsonr


class ProjectionMatrixPredictor(nn.Module):
    """
    預測完整的 2x3 投影矩陣
    
    輸入: 運動/訊號特徵
    輸出: 2x3 投影矩陣 P(t)
    """
    def __init__(self, input_dim=10, hidden_dim=64):
        super(ProjectionMatrixPredictor, self).__init__()
        
        # 特徵提取網路
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU()
        )
        
        # 投影矩陣生成（2x3 = 6 個參數）
        self.projection_head = nn.Linear(32, 6)
        
        # 初始化接近標準 POS
        # P = [[0, 1, -1], [-2, 1, 1]]
        with torch.no_grad():
            self.projection_head.weight.fill_(0.01)
            self.projection_head.bias.copy_(
                torch.tensor([0.0, 1.0, -1.0, -2.0, 1.0, 1.0])
            )
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) 特徵向量
        
        Returns:
            P: (batch, 2, 3) 投影矩陣
        """
        features = self.feature_net(x)
        P_flat = self.projection_head(features)  # (batch, 6)
        P = P_flat.view(-1, 2, 3)  # (batch, 2, 3)
        
        return P


class ConstrainedProjectionPredictor(nn.Module):
    """
    帶約束的投影矩陣預測器
    
    約束條件：
    1. 保持 RGB 平衡（行和約束）
    2. 避免病態矩陣（正則化）
    3. 平滑變化（時序約束）
    """
    def __init__(self, input_dim=10, hidden_dim=64, use_residual=True):
        super(ConstrainedProjectionPredictor, self).__init__()
        
        self.use_residual = use_residual
        
        # 標準 POS 矩陣（作為基準）
        self.register_buffer('P_standard', torch.tensor([
            [0.0, 1.0, -1.0],
            [-2.0, 1.0, 1.0]
        ], dtype=torch.float32))
        
        # 特徵提取
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        if use_residual:
            # 預測殘差（相對於標準 POS 的修正）
            self.residual_head = nn.Linear(32, 6)
            # 小初始化，從標準 POS 開始
            with torch.no_grad():
                self.residual_head.weight.fill_(0.01)
                self.residual_head.bias.fill_(0.0)
        else:
            # 直接預測矩陣
            self.projection_head = nn.Linear(32, 6)
            with torch.no_grad():
                self.projection_head.weight.fill_(0.01)
                self.projection_head.bias.copy_(
                    torch.tensor([0.0, 1.0, -1.0, -2.0, 1.0, 1.0])
                )
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)
        
        Returns:
            P: (batch, 2, 3)
        """
        features = self.feature_net(x)
        
        if self.use_residual:
            # 殘差模式：P(t) = P_standard + ΔP(t)
            residual = self.residual_head(features).view(-1, 2, 3)
            # 限制殘差大小
            residual = torch.tanh(residual) * 0.5  # 最大±0.5的修正
            P = self.P_standard.unsqueeze(0) + residual
        else:
            # 直接預測模式
            P = self.projection_head(features).view(-1, 2, 3)
        
        return P


class TemporalProjectionPredictor(nn.Module):
    """
    時序投影矩陣預測器（使用 LSTM）
    
    考慮時間依賴性，預測平滑的 P(t) 序列
    """
    def __init__(self, window_size=128, hidden_dim=64, num_layers=2):
        super(TemporalProjectionPredictor, self).__init__()
        
        # 標準 POS 矩陣
        self.register_buffer('P_standard', torch.tensor([
            [0.0, 1.0, -1.0],
            [-2.0, 1.0, 1.0]
        ], dtype=torch.float32))
        
        # 1D CNN 提取時序特徵
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM 捕捉時序依賴
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 投影矩陣頭（預測殘差）
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        
        # 初始化
        with torch.no_grad():
            self.projection_head[-1].weight.fill_(0.01)
            self.projection_head[-1].bias.fill_(0.0)
    
    def forward(self, rgb_sequence):
        """
        Args:
            rgb_sequence: (batch, 3, window_size) RGB 時序訊號
        
        Returns:
            P: (batch, 2, 3) 投影矩陣
        """
        # CNN 特徵提取
        features = self.conv(rgb_sequence).squeeze(-1)  # (batch, 32)
        features = features.unsqueeze(1)  # (batch, 1, 32) for LSTM
        
        # LSTM 處理
        lstm_out, _ = self.lstm(features)  # (batch, 1, hidden_dim)
        lstm_out = lstm_out.squeeze(1)  # (batch, hidden_dim)
        
        # 預測殘差
        residual = self.projection_head(lstm_out).view(-1, 2, 3)
        residual = torch.tanh(residual) * 0.5
        
        # 加上標準矩陣
        P = self.P_standard.unsqueeze(0) + residual
        
        return P


class LearnableProjectionPOS:
    """
    可學習投影矩陣的 POS 演算法
    """
    def __init__(self, window_length=128, fs=30):
        self.window_length = window_length
        self.fs = fs
    
    def extract_window_features(self, r_win, g_win, b_win):
        """提取特徵（與之前相同）"""
        features = np.zeros(10)
        
        # RGB 方差
        features[0] = np.var(r_win)
        features[1] = np.var(g_win)
        features[2] = np.var(b_win)
        
        # RGB 梯度
        features[3] = np.mean(np.abs(np.diff(r_win)))
        features[4] = np.mean(np.abs(np.diff(g_win)))
        features[5] = np.mean(np.abs(np.diff(b_win)))
        
        # 歸一化均值
        features[6] = np.mean(r_win) / (np.mean([r_win, g_win, b_win]) + 1e-8)
        features[7] = np.mean(g_win) / (np.mean([r_win, g_win, b_win]) + 1e-8)
        features[8] = np.mean(b_win) / (np.mean([r_win, g_win, b_win]) + 1e-8)
        
        # 高頻能量
        fft_r = np.abs(np.fft.fft(r_win))
        features[9] = np.sum(fft_r[len(fft_r)//2:]) / (np.sum(fft_r) + 1e-8)
        
        return features
    
    def process_standard(self, r_buf, g_buf, b_buf):
        """標準 POS（固定矩陣）"""
        frameNum = len(r_buf)
        l = self.window_length
        
        # 標準投影矩陣
        P = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float32)
        
        S = np.zeros((2, frameNum))
        H = np.zeros(frameNum)
        
        mu_r = np.mean(r_buf)
        mu_g = np.mean(g_buf)
        mu_b = np.mean(b_buf)
        
        for n in range(frameNum):
            m = n - (l - 1)
            if m >= 0:
                # 標準化
                rgb_norm = np.vstack([
                    r_buf[m:n+1]/mu_r - 1,
                    g_buf[m:n+1]/mu_g - 1,
                    b_buf[m:n+1]/mu_b - 1
                ])
                
                # 投影
                S[:, m:n+1] = P @ rgb_norm
                
                # POS 組合（標準 alpha）
                alpha = np.std(S[0, m:n+1]) / (np.std(S[1, m:n+1]) + 1e-8)
                h = S[0, m:n+1] + alpha * S[1, m:n+1]
                h = h - np.mean(h)
                
                # 重疊相加
                H[m:n+1] = H[m:n+1] + h
        
        rppg = -1 * H
        return rppg
    
    def process_learnable(self, r_buf, g_buf, b_buf, model, use_features=True):
        """
        可學習投影矩陣的 POS
        
        Args:
            r_buf, g_buf, b_buf: RGB traces
            model: 訓練好的投影矩陣預測器
            use_features: True=使用特徵, False=使用 RGB 序列
        """
        frameNum = len(r_buf)
        l = self.window_length
        
        H = np.zeros(frameNum)
        P_history = []  # 記錄投影矩陣變化
        
        mu_r = np.mean(r_buf)
        mu_g = np.mean(g_buf)
        mu_b = np.mean(b_buf)
        
        model.eval()
        with torch.no_grad():
            for n in range(frameNum):
                m = n - (l - 1)
                if m >= 0:
                    # 標準化
                    rgb_norm = np.vstack([
                        r_buf[m:n+1]/mu_r - 1,
                        g_buf[m:n+1]/mu_g - 1,
                        b_buf[m:n+1]/mu_b - 1
                    ])
                    
                    # 預測投影矩陣 P(t)
                    if use_features and hasattr(model, 'feature_net'):
                        # 基於特徵
                        features = self.extract_window_features(
                            r_buf[m:n+1], g_buf[m:n+1], b_buf[m:n+1]
                        )
                        features_tensor = torch.FloatTensor(features).unsqueeze(0)
                        P_t = model(features_tensor).squeeze(0).numpy()  # (2, 3)
                    else:
                        # 基於時序
                        rgb_seq = torch.FloatTensor(rgb_norm).unsqueeze(0)  # (1, 3, len)
                        P_t = model(rgb_seq).squeeze(0).numpy()  # (2, 3)
                    
                    P_history.append(P_t.copy())
                    
                    # 使用預測的 P(t) 投影
                    S = P_t @ rgb_norm  # (2, window_length)
                    
                    # POS 組合（標準 alpha）
                    alpha = np.std(S[0, :]) / (np.std(S[1, :]) + 1e-8)
                    h = S[0, :] + alpha * S[1, :]
                    h = h - np.mean(h)
                    
                    # 重疊相加
                    H[m:n+1] = H[m:n+1] + h
        
        rppg = -1 * H
        return rppg, np.array(P_history)


class ProjectionMatrixLoss(nn.Module):
    """
    投影矩陣訓練的損失函數
    
    設計總覽：物理導向的複合目標
    
    L_Total = λ₁·L_Time + λ₂·L_Freq + λ₃·L_Ortho
    
    其中：
    - L_Time: 時域損失 (負皮爾森相關係數) - 主導目標
    - L_Freq: 頻域損失 (頻譜幅度 MSE) - 生理約束  
    - L_Ortho: 正交約束 (幾何結構) - 維持 POS 物理本質
    """
    def __init__(self, lambda_time=1.0, lambda_freq=0.5, lambda_ortho=0.05, fs=30):
        super(ProjectionMatrixLoss, self).__init__()
        self.lambda_time = lambda_time      # 時域權重 (主導，設為 1.0)
        self.lambda_freq = lambda_freq      # 頻域權重 (0.3~0.7)
        self.lambda_ortho = lambda_ortho    # 正交約束 (0.01~0.1，輕微約束)
        self.fs = fs                        # 採樣率
        
        # 人類心率範圍 (0.7-3.0 Hz，對應 42-180 bpm)
        self.hr_range = (0.7, 3.0)
    
    def time_loss(self, pred_rppg, gt_ppg):
        """
        第一部分：時域損失 (Time-Domain Loss) —— 主導目標
        
        負皮爾森相關係數損失 (Negative Pearson Correlation Loss)
        L_Time = 1 - Pearson(H_pred, H_gt)
        
        為什麼不用 MSE？
        - rPPG 訊號的絕對振幅和相位延遲是未知的
        - MSE 嚴厲懲罰振幅差異，導致難以收斂
        - 皮爾森相關關注波形趨勢一致性，對振幅縮放和線性偏移不敏感
        
        功能角色：「導航員」- 指引模型輸出波形走向正確
        
        Args:
            pred_rppg: (batch, signal_length) 預測的 rPPG
            gt_ppg: (batch, signal_length) ground truth PPG
        
        Returns:
            loss: 標量，範圍 [0, 2]
        """
        # 計算均值
        pred_mean = torch.mean(pred_rppg, dim=-1, keepdim=True)
        gt_mean = torch.mean(gt_ppg, dim=-1, keepdim=True)
        
        # 計算標準差
        pred_std = torch.std(pred_rppg, dim=-1, keepdim=True) + 1e-8
        gt_std = torch.std(gt_ppg, dim=-1, keepdim=True) + 1e-8
        
        # 皮爾森相關係數
        correlation = torch.mean(
            (pred_rppg - pred_mean) * (gt_ppg - gt_mean) / (pred_std * gt_std),
            dim=-1
        )
        
        # 負相關損失：最小化損失 = 最大化相關性
        time_loss = 1 - correlation.mean()
        
        return time_loss
    
    def frequency_loss(self, pred_rppg, gt_ppg):
        """
        第二部分：頻域損失 (Frequency-Domain Loss) —— 生理約束
        
        頻譜幅度 MSE 損失 (Spectral Magnitude MSE Loss)
        L_Freq = MSE(|FFT(H_pred)|, |FFT(H_gt)|)
        
        物理意義：
        1. 稀疏性：乾淨的 PPG 訊號在頻域是稀疏的，能量集中在心率基頻和諧波
        2. 抗噪性：運動雜訊通常是寬頻或極低頻
        3. 對齊容忍：只關心心率峰值位置，對相位對齊要求較低
        
        功能角色：「生理學家」- 強迫模型輸出符合人類心率特徵的訊號
        
        Args:
            pred_rppg: (batch, signal_length)
            gt_ppg: (batch, signal_length)
        
        Returns:
            loss: 標量
        """
        # 計算 FFT
        pred_fft = torch.fft.rfft(pred_rppg, dim=-1)  # (batch, freq_bins)
        gt_fft = torch.fft.rfft(gt_ppg, dim=-1)
        
        # 幅度譜
        pred_magnitude = torch.abs(pred_fft)
        gt_magnitude = torch.abs(gt_fft)
        
        # 歸一化（避免數值過大）
        pred_magnitude_norm = pred_magnitude / (torch.sum(pred_magnitude, dim=-1, keepdim=True) + 1e-8)
        gt_magnitude_norm = gt_magnitude / (torch.sum(gt_magnitude, dim=-1, keepdim=True) + 1e-8)
        
        # 基本頻譜 MSE（歸一化後）
        freq_loss_basic = torch.mean((pred_magnitude_norm - gt_magnitude_norm) ** 2)
        
        # 進階：加入頻率遮罩（可選，嚴厲懲罰心率範圍外的能量）
        n_fft = pred_magnitude.shape[-1]
        freq_bins = torch.linspace(0, self.fs/2, n_fft, device=pred_rppg.device)
        
        # 心率範圍內的遮罩
        hr_mask = (freq_bins >= self.hr_range[0]) & (freq_bins <= self.hr_range[1])
        
        # 心率範圍內的頻譜應該更準確
        if hr_mask.sum() > 0:
            freq_loss_hr = torch.mean(
                ((pred_magnitude_norm[:, hr_mask] - gt_magnitude_norm[:, hr_mask]) ** 2)
            )
            # 組合：基本 MSE + 心率範圍加權
            freq_loss = 0.7 * freq_loss_basic + 0.3 * freq_loss_hr
        else:
            freq_loss = freq_loss_basic
        
        return freq_loss
    
    def orthogonality_loss(self, P_batch):
        """
        第三部分：幾何約束損失 (Geometric Constraint Loss) —— 結構守護者
        
        正交性損失 (Orthogonality Loss)
        L_Ortho = Mean((p₁ · p₂)²)
        
        物理意義與必要性：
        - POS 演算法核心假設：投影到兩個正交子空間 (S₁ 和 S₂)
        - 兩個子空間包含不同資訊（類似用兩個不同角度的攝影機）
        - 最後組合來抵銷雜訊
        
        如果沒有這個 Loss：
        - AI 走捷徑，輸出兩個幾乎平行的向量 (p₁ ≈ p₂)
        - S₁(t) 和 S₂(t) 高度相關
        - POS 去噪機制崩潰失效
        - 模型退化成單一投影
        
        功能角色：「結構工程師」- 確保矩陣結構符合 POS 幾何假設
        
        Args:
            P_batch: (batch, 2, 3) 投影矩陣
                p₁ = P_batch[:, 0, :]  第一個投影向量
                p₂ = P_batch[:, 1, :]  第二個投影向量
        
        Returns:
            loss: 標量，理想情況接近 0
        """
        # 提取兩個投影向量
        p1 = P_batch[:, 0, :]  # (batch, 3)
        p2 = P_batch[:, 1, :]  # (batch, 3)
        
        # 計算內積 (如果正交，p₁ · p₂ = 0)
        dot_product = torch.sum(p1 * p2, dim=1)  # (batch,)
        
        # 懲罰非零內積（平方確保正負都被懲罰）
        ortho_loss = torch.mean(dot_product ** 2)
        
        return ortho_loss
    
    def forward(self, pred_rppg, gt_ppg, P_batch):
        """
        總損失計算
        
        L_Total = λ₁·L_Time + λ₂·L_Freq + λ₃·L_Ortho
        
        Args:
            pred_rppg: (batch, signal_length) 預測的 rPPG 訊號
            gt_ppg: (batch, signal_length) ground truth PPG
            P_batch: (batch, 2, 3) 投影矩陣
        
        Returns:
            total_loss: 標量
            metrics: dict，包含各項損失的數值
        """
        # 1. 時域損失（主導目標）
        loss_time = self.time_loss(pred_rppg, gt_ppg)
        
        # 2. 頻域損失（生理約束）
        loss_freq = self.frequency_loss(pred_rppg, gt_ppg)
        
        # 3. 正交約束（結構守護）
        loss_ortho = self.orthogonality_loss(P_batch)
        
        # 總損失
        total_loss = (
            self.lambda_time * loss_time +
            self.lambda_freq * loss_freq +
            self.lambda_ortho * loss_ortho
        )
        
        # 返回詳細指標
        return total_loss, {
            'time': loss_time.item(),
            'freq': loss_freq.item(),
            'ortho': loss_ortho.item(),
            'total': total_loss.item()
        }


if __name__ == "__main__":
    print("Learnable Projection Matrix POS (LP-POS)")
    print("=" * 60)
    
    # 測試模型
    print("\n1. 測試投影矩陣預測器")
    model1 = ProjectionMatrixPredictor(input_dim=10, hidden_dim=64)
    print(f"   參數量: {sum(p.numel() for p in model1.parameters())}")
    
    # 測試前向傳播
    features = torch.randn(4, 10)
    P_pred = model1(features)
    print(f"   輸出形狀: {P_pred.shape}")  # (4, 2, 3)
    print(f"   標準 POS: [[0, 1, -1], [-2, 1, 1]]")
    print(f"   預測範例:\n{P_pred[0]}")
    
    print("\n2. 測試約束投影預測器")
    model2 = ConstrainedProjectionPredictor(input_dim=10, hidden_dim=64, use_residual=True)
    print(f"   參數量: {sum(p.numel() for p in model2.parameters())}")
    P_pred2 = model2(features)
    print(f"   預測範例 (residual模式):\n{P_pred2[0]}")
    
    print("\n3. 測試時序預測器")
    model3 = TemporalProjectionPredictor(window_size=128, hidden_dim=64)
    print(f"   參數量: {sum(p.numel() for p in model3.parameters())}")
    rgb_seq = torch.randn(4, 3, 128)
    P_pred3 = model3(rgb_seq)
    print(f"   輸出形狀: {P_pred3.shape}")
    
    print("\n4. 測試處理器")
    pos = LearnableProjectionPOS(window_length=128, fs=30)
    
    # 生成測試數據
    n_samples = 500
    r_buf = 100 + 5 * np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples)
    g_buf = 120 + 6 * np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples)
    b_buf = 80 + 3 * np.sin(np.arange(n_samples) * 0.1) + np.random.randn(n_samples)
    
    # 標準 POS
    rppg_std = pos.process_standard(r_buf, g_buf, b_buf)
    print(f"   標準 POS 輸出長度: {len(rppg_std)}")
    
    # 可學習 POS
    rppg_learn, P_hist = pos.process_learnable(r_buf, g_buf, b_buf, model1)
    print(f"   可學習 POS 輸出長度: {len(rppg_learn)}")
    print(f"   投影矩陣歷史: {P_hist.shape}")
    
    print("\n5. 測試損失函數")
    criterion = ProjectionMatrixLoss(lambda_ortho=0.1, lambda_smooth=0.1, lambda_reg=0.01)
    
    pred_rppg = torch.randn(4, 300)
    gt_ppg = torch.randn(4, 300)
    P_batch = P_pred
    P_standard = torch.tensor([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    
    loss, metrics = criterion(pred_rppg, gt_ppg, P_batch, P_standard=P_standard)
    print(f"   總損失: {loss.item():.4f}")
    print(f"   詳細指標: {metrics}")
    
    print("\n✓ 所有測試完成！")