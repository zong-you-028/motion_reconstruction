"""
可學習投影矩陣 POS - 完整整合版本
包含：數據處理、訓練、評估、可視化
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# 導入自定義模組
from learnable_projection_pos import (
    ConstrainedProjectionPredictor,
    ProjectionMatrixPredictor,
    TemporalProjectionPredictor,
    LearnableProjectionPOS,
    ProjectionMatrixLoss
)
from data_loader import DataLoader
# 修改這裡：使用正確的函數名稱並取別名，同時導入 Dataset 類別
from train_projection_pos import train_projection_model as train_core, RPPGDatasetForProjection


# ============================================================================
#                           1. 批量數據處理
# ============================================================================

def batch_process_nas_data(nas_root, local_output, frames_folder="cam0", 
                           ppg_filename="PPG_CMS50E_30fps.csv", subjects=None):
    """
    批量處理 NAS 上的數據
    """
    print("\n" + "="*80)
    print("批量處理 NAS 數據")
    print("="*80)
    
    nas_path = Path(nas_root)
    local_path = Path(local_output)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # 掃描受試者
    if subjects is None:
        print(f"\n正在掃描: {nas_root}")
        subjects = []
        for item in nas_path.iterdir():
            if item.is_dir():
                frames_dir = item / frames_folder
                ppg_file = item / ppg_filename
                if frames_dir.exists() and ppg_file.exists():
                    subjects.append(item.name)
                    print(f"  ✓ {item.name}")
                else:
                    print(f"  ✗ {item.name} (缺少檔案)")
        subjects = sorted(subjects)
        print(f"\n找到 {len(subjects)} 個有效受試者")
    
    if not subjects:
        print("✗ 未找到任何有效受試者")
        return 0
    
    # 確認
    print(f"\n將處理 {len(subjects)} 個受試者")
    confirm = input("確認？(y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消")
        return 0
    
    # 批量處理
    success_count = 0
    failed_subjects = []
    
    for i, subject_id in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] 處理 {subject_id}...")
        
        try:
            # 路徑設定
            nas_frames = nas_path / subject_id / frames_folder
            nas_ppg = nas_path / subject_id / ppg_filename
            local_subject = local_path / subject_id
            local_subject.mkdir(parents=True, exist_ok=True)
            
            # 步驟 1: 提取 ROI
            print("  步驟 1: 提取 ROI")
            result = subprocess.run([
                sys.executable,
                "facemesh_roi_cheeks_only.py",
                str(nas_frames),
                subject_id,
                str(local_output)
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # 解析並保存 CSV
                output_lines = result.stdout.strip().split('\n')
                data_lines = []
                for line in output_lines:
                    if ',' in line and not line.startswith('frame'):
                        try:
                            parts = line.split(',')
                            if len(parts) == 5:
                                float(parts[0])
                                data_lines.append(line)
                        except:
                            continue
                
                if data_lines:
                    csv_path = local_subject / f"{subject_id}_rgb_traces.csv"
                    with open(csv_path, 'w') as f:
                        f.write("frame,R_avg,G_avg,B_avg,success\n")
                        f.write('\n'.join(data_lines))
                    print(f"    ✓ RGB CSV: {len(data_lines)} 幀")
                else:
                    print(f"    ✗ 未獲取到有效數據")
                    failed_subjects.append(subject_id)
                    continue
            else:
                print(f"    ✗ ROI 提取失敗")
                if result.stderr:
                    print(f"    錯誤: {result.stderr[:200]}")
                failed_subjects.append(subject_id)
                continue
            
            # 步驟 2: 複製 PPG
            print("  步驟 2: 複製 PPG")
            local_ppg = local_subject / "ppg.csv"
            shutil.copy(nas_ppg, local_ppg)
            print(f"    ✓ PPG 已複製")
            
            # 步驟 3: 驗證
            rgb_csv = local_subject / f"{subject_id}_rgb_traces.csv"
            if rgb_csv.exists() and local_ppg.exists():
                print(f"  ✅ {subject_id} 處理完成")
                success_count += 1
            else:
                print(f"  ❌ {subject_id} 驗證失敗")
                failed_subjects.append(subject_id)
                
        except Exception as e:
            print(f"  ✗ 異常: {e}")
            failed_subjects.append(subject_id)
    
    # 總結
    print("\n" + "="*80)
    print(f"處理完成: 成功 {success_count}/{len(subjects)}")
    if failed_subjects:
        print(f"失敗的受試者: {', '.join(failed_subjects)}")
    print("="*80)
    
    return success_count


# ============================================================================
#                           2. 數據診斷
# ============================================================================

def diagnose_subject_data(subject_dir):
    """
    診斷單個受試者的數據質量
    """
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name
    
    result = {
        'subject_id': subject_id,
        'is_valid': True,
        'issues': []
    }
    
    # 檢查檔案
    rgb_csv = list(subject_dir.glob('*rgb_traces.csv'))
    ppg_csv = list(subject_dir.glob('ppg.csv')) + list(subject_dir.glob('PPG*.csv'))
    
    if not rgb_csv or not ppg_csv:
        result['is_valid'] = False
        result['issues'].append("缺少必要檔案")
        return result
    
    try:
        # RGB 檢查
        df_rgb = pd.read_csv(rgb_csv[0])
        
        if 'success' in df_rgb.columns:
            success_rate = (df_rgb['success'] == 1).sum() / len(df_rgb)
            if success_rate < 0.5:
                result['is_valid'] = False
                result['issues'].append(f"RGB 成功率過低 ({success_rate*100:.1f}%)")
        
        for ch in ['R_avg', 'G_avg', 'B_avg']:
            if ch in df_rgb.columns and df_rgb[ch].std() < 1:
                result['is_valid'] = False
                result['issues'].append(f"{ch} 變化過小")
        
        # PPG 檢查
        try:
            ppg = pd.read_csv(ppg_csv[0])
            ppg_values = ppg.iloc[:, 0].values
        except:
            ppg = pd.read_csv(ppg_csv[0], header=None)
            ppg_values = ppg[0].values
        
        if len(np.unique(ppg_values)) < 10:
            result['is_valid'] = False
            result['issues'].append("PPG 是常數或接近常數")
        
        if ppg_values.std() < 1:
            result['is_valid'] = False
            result['issues'].append("PPG 標準差過小")
            
    except Exception as e:
        result['is_valid'] = False
        result['issues'].append(f"讀取錯誤: {e}")
    
    return result


def diagnose_all_subjects(data_dir):
    """診斷所有受試者"""
    print("\n" + "="*80)
    print("診斷所有受試者數據")
    print("="*80)
    
    data_path = Path(data_dir)
    subject_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    valid_subjects = []
    invalid_subjects = []
    
    for subject_dir in sorted(subject_dirs):
        result = diagnose_subject_data(subject_dir)
        
        if result['is_valid']:
            print(f"  ✓ {result['subject_id']}")
            valid_subjects.append(result['subject_id'])
        else:
            print(f"  ✗ {result['subject_id']}: {', '.join(result['issues'])}")
            invalid_subjects.append(result)
    
    print("\n" + "="*80)
    print(f"有效受試者: {len(valid_subjects)}")
    print(f"無效受試者: {len(invalid_subjects)}")
    
    if invalid_subjects:
        print("\n建議排除以下受試者:")
        for r in invalid_subjects:
            print(f"  - {r['subject_id']}: {', '.join(r['issues'])}")
    
    print("="*80)
    
    return valid_subjects, invalid_subjects


# ============================================================================
#                           3. 模型訓練
# ============================================================================

def train_projection_model(data_dir, output_dir, model_type=1, epochs=50, 
                           batch_size=32, learning_rate=0.001, device=None):
    """
    訓練投影矩陣模型
    """
    print("\n" + "="*80)
    print("訓練投影矩陣模型")
    print("="*80)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n配置:")
    print(f"  數據目錄: {data_dir}")
    print(f"  輸出目錄: {output_dir}")
    print(f"  模型類型: {model_type}")
    print(f"  訓練輪數: {epochs}")
    print(f"  設備: {device}")
    
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 載入數據
    print("\n正在載入數據...")
    data_path = Path(data_dir)
    
    # 診斷並過濾無效數據
    valid_subjects, invalid_subjects = diagnose_all_subjects(data_dir)
    
    if len(valid_subjects) < 3:
        print(f"\n✗ 有效受試者太少 ({len(valid_subjects)})，至少需要 3 個")
        return None
    
    # 載入有效受試者的數據
    loader = DataLoader(data_dir=str(data_path), fs=84)
    
    rgb_traces = []
    ppg_signals = []
    
    for subject_id in valid_subjects:
        subject_dir = data_path / subject_id
        rgb_csv = list(subject_dir.glob('*rgb_traces.csv'))
        ppg_csv = list(subject_dir.glob('ppg.csv')) + list(subject_dir.glob('PPG*.csv'))
        
        if rgb_csv and ppg_csv:
            try:
                data = loader.load_subject_data(
                    subject_id=subject_id,
                    rgb_csv_path=str(rgb_csv[0]),
                    ppg_path=str(ppg_csv[0]),
                    rgb_start_frame=0
                )
                rgb_traces.append((data['r'], data['g'], data['b']))
                ppg_signals.append(data['ppg'])
                print(f"  ✓ {subject_id}: {len(data['r'])} 樣本")
            except Exception as e:
                print(f"  ✗ {subject_id}: {e}")
    
    if len(rgb_traces) == 0:
        print("\n✗ 未能載入任何數據")
        return None
    
    print(f"\n✓ 成功載入 {len(rgb_traces)} 個受試者")
    
    # 分割數據與建立 Dataset/DataLoader
    print("\n正在準備訓練數據 (Dataset & DataLoader)...")
    split_idx = int(0.8 * len(rgb_traces))
    train_rgb = rgb_traces[:split_idx]
    train_ppg = ppg_signals[:split_idx]
    val_rgb = rgb_traces[split_idx:]
    val_ppg = ppg_signals[split_idx:]

    # 根據模型類型決定 Dataset 模式
    # Model type 1 & 2 -> 'feature', Model type 3 -> 'sequence'
    dataset_mode = 'sequence' if model_type == 3 else 'feature'
    print(f"  Dataset 模式: {dataset_mode}")

    # 創建 Dataset
    train_dataset = RPPGDatasetForProjection(
        train_rgb, train_ppg, window_length=128, stride=32, mode=dataset_mode
    )
    val_dataset = RPPGDatasetForProjection(
        val_rgb, val_ppg, window_length=128, stride=64, mode=dataset_mode
    )
    
    # 創建 DataLoader
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  訓練 Batch 數: {len(train_loader)}")
    print(f"  驗證 Batch 數: {len(val_loader)}")

    # 創建模型
    print("\n正在創建模型...")
    
    if model_type == 1:
        model = ConstrainedProjectionPredictor(
            input_dim=10, 
            hidden_dim=64, 
            use_residual=True
        )
        model_name = "ConstrainedProjectionPredictor"
    elif model_type == 2:
        model = ProjectionMatrixPredictor(
            input_dim=10, 
            hidden_dim=64
        )
        model_name = "ProjectionMatrixPredictor"
    elif model_type == 3:
        model = TemporalProjectionPredictor(
            window_size=128, 
            hidden_dim=64
        )
        model_name = "TemporalProjectionPredictor"
    else:
        print(f"✗ 無效的模型類型: {model_type}")
        return None
    
    print(f"  模型: {model_name}")
    print(f"  參數量: {sum(p.numel() for p in model.parameters())}")
    
    # 訓練
    print("\n開始訓練...")
    
    # 使用別名 train_core 並傳入 DataLoader
    model, train_losses, val_losses = train_core(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=str(output_path),
        model_type=dataset_mode
    )
    
    best_model_path = str(output_path / 'best_projection_model.pth')
    
    print(f"\n✓ 訓練完成")
    print(f"  最佳模型路徑: {best_model_path}")
    
    return best_model_path


# ============================================================================
#                           4. 模型評估
# ============================================================================

def calculate_hr_from_rppg(rppg_signal, fs):
    """計算心率"""
    n = len(rppg_signal)
    fft_vals = np.fft.fft(rppg_signal)
    fft_freq = np.fft.fftfreq(n, 1/fs)
    
    pos_mask = (fft_freq > 0) & (fft_freq < 5)
    fft_freq = fft_freq[pos_mask]
    fft_vals = np.abs(fft_vals[pos_mask])
    
    hr_mask = (fft_freq >= 0.7) & (fft_freq <= 4.0)
    
    if not np.any(hr_mask):
        return 0
    
    hr_freq = fft_freq[hr_mask]
    hr_fft = fft_vals[hr_mask]
    
    peak_idx = np.argmax(hr_fft)
    hr_hz = hr_freq[peak_idx]
    hr_bpm = hr_hz * 60
    
    return hr_bpm


def evaluate_single_subject(model, rgb_traces, ppg_gt, subject_id, fs=84, device='cpu'):
    """評估單個受試者"""
    r, g, b = rgb_traces
    
    pos = LearnableProjectionPOS(window_length=128, fs=fs)
    
    # 標準 POS
    rppg_standard = pos.process_standard(r, g, b)
    
    # 可學習 POS
    model.eval()
    with torch.no_grad():
        rppg_learned, P_history = pos.process_learnable(r, g, b, model, use_features=True)
    
    # 去除初始段
    valid_start = 128
    rppg_standard = rppg_standard[valid_start:]
    rppg_learned = rppg_learned[valid_start:]
    ppg_gt = ppg_gt[valid_start:]
    
    # 確保長度一致
    min_len = min(len(rppg_standard), len(rppg_learned), len(ppg_gt))
    rppg_standard = rppg_standard[:min_len]
    rppg_learned = rppg_learned[:min_len]
    ppg_gt = ppg_gt[:min_len]
    
    # 帶通濾波
    nyq = fs / 2
    b_filt, a_filt = signal.butter(4, [0.7/nyq, 4.0/nyq], btype='band')
    
    rppg_std_filt = signal.filtfilt(b_filt, a_filt, rppg_standard)
    rppg_learn_filt = signal.filtfilt(b_filt, a_filt, rppg_learned)
    ppg_gt_filt = signal.filtfilt(b_filt, a_filt, ppg_gt)
    
    # 計算指標
    corr_std, _ = pearsonr(rppg_std_filt, ppg_gt_filt)
    corr_learn, _ = pearsonr(rppg_learn_filt, ppg_gt_filt)
    
    # SNR
    def calculate_snr(sig, fs):
        n = len(sig)
        fft_vals = np.fft.fft(sig)
        fft_freq = np.fft.fftfreq(n, 1/fs)
        
        pos_mask = fft_freq > 0
        fft_vals = np.abs(fft_vals[pos_mask])
        fft_freq = fft_freq[pos_mask]
        
        hr_mask = (fft_freq >= 0.7) & (fft_freq <= 4.0)
        signal_power = np.sum(fft_vals[hr_mask] ** 2)
        noise_power = np.sum(fft_vals[~hr_mask] ** 2)
        
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return 0
    
    snr_std = calculate_snr(rppg_std_filt, fs)
    snr_learn = calculate_snr(rppg_learn_filt, fs)
    
    # 心率
    hr_gt = calculate_hr_from_rppg(ppg_gt_filt, fs)
    hr_std = calculate_hr_from_rppg(rppg_std_filt, fs)
    hr_learn = calculate_hr_from_rppg(rppg_learn_filt, fs)
    
    return {
        'subject_id': subject_id,
        'metrics': {
            'corr_std': corr_std,
            'corr_learn': corr_learn,
            'snr_std': snr_std,
            'snr_learn': snr_learn,
            'hr_gt': hr_gt,
            'hr_std': hr_std,
            'hr_learn': hr_learn,
            'hr_error_std': abs(hr_std - hr_gt),
            'hr_error_learn': abs(hr_learn - hr_gt)
        }
    }


def evaluate_model(model_path, data_dir, output_dir, device=None):
    """評估模型"""
    print("\n" + "="*80)
    print("模型評估")
    print("="*80)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n配置:")
    print(f"  模型: {model_path}")
    print(f"  數據: {data_dir}")
    print(f"  輸出: {output_dir}")
    print(f"  設備: {device}")
    
    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 載入模型
    print("\n正在載入模型...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = ConstrainedProjectionPredictor(input_dim=10, hidden_dim=64, use_residual=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("  ✓ 模型載入完成")
    
    # 診斷並載入數據
    print("\n正在載入數據...")
    valid_subjects, invalid_subjects = diagnose_all_subjects(data_dir)
    
    if len(valid_subjects) == 0:
        print("✗ 沒有有效的受試者數據")
        return
    
    print(f"\n將評估 {len(valid_subjects)} 個受試者")
    
    # 載入並評估
    data_path = Path(data_dir)
    loader = DataLoader(data_dir=str(data_path), fs=84)
    
    all_results = []
    
    for subject_id in valid_subjects:
        subject_dir = data_path / subject_id
        rgb_csv = list(subject_dir.glob('*rgb_traces.csv'))
        ppg_csv = list(subject_dir.glob('ppg.csv')) + list(subject_dir.glob('PPG*.csv'))
        
        if not rgb_csv or not ppg_csv:
            continue
        
        try:
            print(f"\n處理 {subject_id}...")
            
            # 載入數據
            data = loader.load_subject_data(
                subject_id=subject_id,
                rgb_csv_path=str(rgb_csv[0]),
                ppg_path=str(ppg_csv[0]),
                rgb_start_frame=0
            )
            
            rgb_traces = (data['r'], data['g'], data['b'])
            ppg_gt = data['ppg']
            
            # 評估
            results = evaluate_single_subject(
                model, rgb_traces, ppg_gt, subject_id, fs=84, device=device
            )
            
            all_results.append(results)
            
            # 顯示指標
            m = results['metrics']
            print(f"  Correlation: {m['corr_std']:.4f} → {m['corr_learn']:.4f} ({(m['corr_learn']-m['corr_std'])/abs(m['corr_std'])*100:+.2f}%)")
            print(f"  SNR: {m['snr_std']:.2f} → {m['snr_learn']:.2f} dB ({m['snr_learn']-m['snr_std']:+.2f} dB)")
            print(f"  HR Error: {m['hr_error_std']:.2f} → {m['hr_error_learn']:.2f} bpm")
            
        except Exception as e:
            print(f"  ✗ 評估失敗: {e}")
            continue
    
    # 保存結果
    if all_results:
        results_csv = output_path / "evaluation_metrics.csv"
        metrics_data = []
        for r in all_results:
            m = r['metrics']
            metrics_data.append({
                'Subject': r['subject_id'],
                'Corr_Std': m['corr_std'],
                'Corr_Learn': m['corr_learn'],
                'SNR_Std': m['snr_std'],
                'SNR_Learn': m['snr_learn'],
                'HR_GT': m['hr_gt'],
                'HR_Std': m['hr_std'],
                'HR_Learn': m['hr_learn'],
                'HR_Error_Std': m['hr_error_std'],
                'HR_Error_Learn': m['hr_error_learn']
            })
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(results_csv, index=False)
        print(f"\n✓ 數值結果已保存: {results_csv}")
        
        # 顯示統計
        print("\n" + "="*80)
        print("整體統計")
        print("="*80)
        print(f"  Correlation: {df['Corr_Std'].mean():.4f} → {df['Corr_Learn'].mean():.4f} ({(df['Corr_Learn'].mean()-df['Corr_Std'].mean())/abs(df['Corr_Std'].mean())*100:+.2f}%)")
        print(f"  SNR: {df['SNR_Std'].mean():.2f} → {df['SNR_Learn'].mean():.2f} dB ({df['SNR_Learn'].mean()-df['SNR_Std'].mean():+.2f} dB)")
        print(f"  HR Error: {df['HR_Error_Std'].mean():.2f} → {df['HR_Error_Learn'].mean():.2f} bpm")
        print("="*80)
    
    return all_results


# ============================================================================
#                           5. 主選單
# ============================================================================

def print_menu():
    """顯示主選單"""
    print("\n" + "="*80)
    print("可學習投影矩陣 POS - 完整工具集")
    print("="*80)
    print("\n選擇功能:")
    print("  1. 批量處理 NAS 數據（從影片提取 ROI + PPG）")
    print("  2. 診斷數據質量")
    print("  3. 訓練模型")
    print("  4. 評估模型")
    print("  5. 完整流程（數據處理 → 訓練 → 評估）")
    print("  0. 退出")
    print("="*80)


def interactive_mode():
    """互動模式"""
    while True:
        print_menu()
        choice = input("\n請選擇功能 [1-5, 0退出]: ").strip()
        
        if choice == '0':
            print("\n再見！")
            break
            
        elif choice == '1':
            # 批量處理
            print("\n=== 批量處理 NAS 數據 ===")
            nas_root = input("NAS 根目錄 [\\\\10.1.1.3\\bio3\\PURE_dataset]: ").strip()
            if not nas_root:
                nas_root = r"\\10.1.1.3\bio3\PURE_dataset"
            
            local_output = input("本地輸出目錄 [D:\\rppg_output]: ").strip()
            if not local_output:
                local_output = r"D:\rppg\motion_reconstruction\rppg_output"
            
            batch_process_nas_data(nas_root, local_output)
            
        elif choice == '2':
            # 診斷數據
            print("\n=== 診斷數據質量 ===")
            data_dir = input("數據目錄 [D:\\rppg_output]: ").strip()
            if not data_dir:
                data_dir = r"D:\rppg\motion_reconstruction\rppg_output"
            
            diagnose_all_subjects(data_dir)
            
        elif choice == '3':
            # 訓練模型
            print("\n=== 訓練模型 ===")
            data_dir = input("數據目錄 [D:\\rppg_output]: ").strip()
            if not data_dir:
                data_dir = r"D:\rppg\motion_reconstruction\rppg_output"
            
            output_dir = input("輸出目錄 [./projection_models]: ").strip()
            if not output_dir:
                output_dir = "./projection_models"
            
            print("\n選擇模型類型:")
            print("  1. ConstrainedProjectionPredictor (推薦)")
            print("  2. ProjectionMatrixPredictor (基礎)")
            print("  3. TemporalProjectionPredictor (時序)")
            model_type = input("請選擇 [1]: ").strip()
            model_type = int(model_type) if model_type else 1
            
            epochs = input("訓練輪數 [50]: ").strip()
            epochs = int(epochs) if epochs else 50
            
            train_projection_model(
                data_dir=data_dir,
                output_dir=output_dir,
                model_type=model_type,
                epochs=epochs
            )
            
        elif choice == '4':
            # 評估模型
            print("\n=== 評估模型 ===")
            model_path = input("模型路徑: ").strip()
            if not model_path:
                print("✗ 請提供模型路徑")
                continue
            
            data_dir = input("數據目錄 [D:\\rppg_output]: ").strip()
            if not data_dir:
                data_dir = r"D:\rppg\motion_reconstruction\rppg_output"
            
            output_dir = input("輸出目錄 [./evaluation_results]: ").strip()
            if not output_dir:
                output_dir = "./evaluation_results"
            
            evaluate_model(model_path, data_dir, output_dir)
            
        elif choice == '5':
            # 完整流程
            print("\n=== 完整流程 ===")
            
            # 步驟 1: 批量處理
            print("\n步驟 1/3: 批量處理數據")
            do_process = input("是否執行數據處理？(y/n) [y]: ").strip().lower()
            if not do_process or do_process == 'y':
                nas_root = input("NAS 根目錄 [\\\\10.1.1.3\\bio3\\PURE_dataset]: ").strip()
                if not nas_root:
                    nas_root = r"\\10.1.1.3\bio3\PURE_dataset"
                
                local_output = input("本地輸出目錄 [D:\\rppg_output]: ").strip()
                if not local_output:
                    local_output = r"D:\rppg\motion_reconstruction\rppg_output"
                
                success = batch_process_nas_data(nas_root, local_output)
                if success == 0:
                    print("✗ 數據處理失敗")
                    continue
            else:
                local_output = input("現有數據目錄 [D:\\rppg_output]: ").strip()
                if not local_output:
                    local_output = r"D:\rppg\motion_reconstruction\rppg_output"
            
            # 步驟 2: 訓練
            print("\n步驟 2/3: 訓練模型")
            output_dir = input("模型輸出目錄 [./projection_models]: ").strip()
            if not output_dir:
                output_dir = "./projection_models"
            
            epochs = input("訓練輪數 [50]: ").strip()
            epochs = int(epochs) if epochs else 50
            
            model_path = train_projection_model(
                data_dir=local_output,
                output_dir=output_dir,
                model_type=1,
                epochs=epochs
            )
            
            if model_path is None:
                print("✗ 訓練失敗")
                continue
            
            # 步驟 3: 評估
            print("\n步驟 3/3: 評估模型")
            eval_dir = input("評估結果輸出目錄 [./evaluation_results]: ").strip()
            if not eval_dir:
                eval_dir = "./evaluation_results"
            
            evaluate_model(model_path, local_output, eval_dir)
            
            print("\n" + "="*80)
            print("✅ 完整流程執行完畢！")
            print("="*80)
        
        else:
            print("✗ 無效的選擇")
        
        input("\n按 Enter 繼續...")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='可學習投影矩陣 POS - 完整工具集')
    parser.add_argument('--mode', type=str, choices=['batch', 'diagnose', 'train', 'evaluate', 'full'], 
                        help='運行模式')
    parser.add_argument('--nas-root', type=str, help='NAS 根目錄')
    parser.add_argument('--data-dir', type=str, help='數據目錄')
    parser.add_argument('--output-dir', type=str, help='輸出目錄')
    parser.add_argument('--model-path', type=str, help='模型路徑')
    parser.add_argument('--model-type', type=int, choices=[1, 2, 3], default=1, help='模型類型')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    
    args = parser.parse_args()
    
    if args.mode is None:
        # 互動模式
        interactive_mode()
    else:
        # 命令列模式
        if args.mode == 'batch':
            if not args.nas_root or not args.output_dir:
                print("✗ 請提供 --nas-root 和 --output-dir")
                return
            batch_process_nas_data(args.nas_root, args.output_dir)
            
        elif args.mode == 'diagnose':
            if not args.data_dir:
                print("✗ 請提供 --data-dir")
                return
            diagnose_all_subjects(args.data_dir)
            
        elif args.mode == 'train':
            if not args.data_dir or not args.output_dir:
                print("✗ 請提供 --data-dir 和 --output-dir")
                return
            train_projection_model(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                model_type=args.model_type,
                epochs=args.epochs
            )
            
        elif args.mode == 'evaluate':
            if not args.model_path or not args.data_dir or not args.output_dir:
                print("✗ 請提供 --model-path, --data-dir 和 --output-dir")
                return
            evaluate_model(args.model_path, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()