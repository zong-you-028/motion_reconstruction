"""
可學習投影矩陣 POS - 完整整合版本
包含：數據處理、訓練、評估 (呼叫 evaluate_and_plot.py)
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
import warnings
warnings.filterwarnings('ignore')

# 導入自定義模組
from learnable_projection_pos import (
    ConstrainedProjectionPredictor,
    ProjectionMatrixPredictor,
    TemporalProjectionPredictor
)
from data_loader import DataLoader
from train_projection_pos import train_projection_model as train_core, RPPGDatasetForProjection

# 導入評估與繪圖模組
from evaluate_and_plot import (
    evaluate_single_subject,
    plot_single_subject_results,
    plot_overall_summary
)


# ============================================================================
#                           1. 批量數據處理
# ============================================================================

def batch_process_nas_data(nas_root, local_output, frames_folder="cam0", 
                           ppg_filename="PPG_CMS50E_30fps.csv", subjects=None):
    """批量處理 NAS 上的數據"""
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
    
    return success_count


# ============================================================================
#                           2. 數據診斷
# ============================================================================

def diagnose_subject_data(subject_dir):
    """診斷單個受試者的數據質量"""
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
        
        # PPG 檢查
        try:
            ppg = pd.read_csv(ppg_csv[0])
            ppg_values = ppg.iloc[:, 0].values
        except:
            ppg = pd.read_csv(ppg_csv[0], header=None)
            ppg_values = ppg[0].values
        
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
    
    if invalid_subjects:
        print("\n[注意] 以下受試者數據品質可能不佳，但將會被強制納入訓練:")
        for r in invalid_subjects:
            print(f"  - {r['subject_id']}: {', '.join(r['issues'])}")
    
    return valid_subjects, invalid_subjects


# ============================================================================
#                           3. 模型訓練
# ============================================================================

def train_projection_model(data_dir, output_dir, model_type=1, epochs=50, 
                           batch_size=32, learning_rate=0.001, device=None):
    """訓練投影矩陣模型"""
    print("\n" + "="*80)
    print("訓練投影矩陣模型")
    print("="*80)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(data_dir)
    
    # 診斷 (不過濾)
    valid_subjects, invalid_subjects = diagnose_all_subjects(data_dir)
    all_subjects_to_use = sorted(valid_subjects + [r['subject_id'] for r in invalid_subjects])
    
    if len(all_subjects_to_use) < 1:
        print(f"\n✗ 未找到任何受試者")
        return None
    
    print(f"\n將使用 {len(all_subjects_to_use)} 個受試者進行訓練")

    # 載入數據 (fs=30)
    loader = DataLoader(data_dir=str(data_path), fs=30)
    
    rgb_traces = []
    ppg_signals = []
    
    for subject_id in all_subjects_to_use:
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
    
    # 分割數據
    split_idx = int(0.8 * len(rgb_traces))
    train_rgb = rgb_traces[:split_idx]
    train_ppg = ppg_signals[:split_idx]
    val_rgb = rgb_traces[split_idx:]
    val_ppg = ppg_signals[split_idx:]

    dataset_mode = 'sequence' if model_type == 3 else 'feature'
    
    train_dataset = RPPGDatasetForProjection(
        train_rgb, train_ppg, window_length=128, stride=32, mode=dataset_mode
    )
    val_dataset = RPPGDatasetForProjection(
        val_rgb, val_ppg, window_length=128, stride=64, mode=dataset_mode
    )
    
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 創建模型
    if model_type == 1:
        model = ConstrainedProjectionPredictor(input_dim=10, hidden_dim=64, use_residual=True)
    elif model_type == 2:
        model = ProjectionMatrixPredictor(input_dim=10, hidden_dim=64)
    elif model_type == 3:
        model = TemporalProjectionPredictor(window_size=128, hidden_dim=64)
    
    print(f"  模型: {type(model).__name__}")
    
    # 訓練
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
    print(f"\n✓ 訓練完成: {best_model_path}")
    return best_model_path


# ============================================================================
#                           4. 模型評估
# ============================================================================

def evaluate_model(model_path, data_dir, output_dir, device=None):
    """評估模型"""
    print("\n" + "="*80)
    print("模型評估")
    print("="*80)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 載入模型
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ConstrainedProjectionPredictor(input_dim=10, hidden_dim=64, use_residual=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("  ✓ 模型載入完成")
    
    # 載入數據
    valid_subjects, invalid_subjects = diagnose_all_subjects(data_dir)
    all_subjects_to_use = sorted(valid_subjects + [r['subject_id'] for r in invalid_subjects])
    
    if len(all_subjects_to_use) == 0:
        print("✗ 沒有找到任何受試者數據")
        return
    
    loader = DataLoader(data_dir=str(data_dir), fs=30)
    all_results = []
    
    for subject_id in all_subjects_to_use:
        subject_dir = Path(data_dir) / subject_id
        rgb_csv = list(subject_dir.glob('*rgb_traces.csv'))
        ppg_csv = list(subject_dir.glob('ppg.csv')) + list(subject_dir.glob('PPG*.csv'))
        
        if not rgb_csv or not ppg_csv:
            continue
        
        try:
            print(f"\n處理 {subject_id}...")
            data = loader.load_subject_data(
                subject_id=subject_id,
                rgb_csv_path=str(rgb_csv[0]),
                ppg_path=str(ppg_csv[0]),
                rgb_start_frame=0
            )
            
            rgb_traces = (data['r'], data['g'], data['b'])
            ppg_gt = data['ppg']
            
            # 使用 evaluate_and_plot 模組進行評估
            results = evaluate_single_subject(
                model, rgb_traces, ppg_gt, subject_id, fs=30, device=device
            )
            all_results.append(results)
            
            # 繪圖
            plot_path = output_path / f"{subject_id}_evaluation.png"
            plot_single_subject_results(results, plot_path)
            
            m = results['metrics']
            print(f"  Corr: {m['corr_std']:.3f} -> {m['corr_learn']:.3f}")
            
        except Exception as e:
            print(f"  ✗ 評估失敗: {e}")
            continue
    
    # 總結圖
    if all_results:
        summary_path = output_path / "overall_summary.png"
        plot_overall_summary(all_results, summary_path)
        
        # 保存 CSV
        results_csv = output_path / "evaluation_metrics.csv"
        metrics_data = []
        for r in all_results:
            m = r['metrics']
            metrics_data.append({
                'Subject': r['subject_id'],
                'Corr_Std': m['corr_std'], 'Corr_Learn': m['corr_learn'],
                'SNR_Std': m['snr_std'], 'SNR_Learn': m['snr_learn'],
                'HR_Error_Std': m['hr_error_std'], 'HR_Error_Learn': m['hr_error_learn']
            })
        pd.DataFrame(metrics_data).to_csv(results_csv, index=False)
        print(f"\n  ✓ 結果已保存: {results_csv}")


# ============================================================================
#                           5. 主選單
# ============================================================================

def interactive_mode():
    """互動模式"""
    while True:
        print("\n" + "="*80)
        print("可學習投影矩陣 POS - 完整工具集")
        print("="*80)
        print("  1. 批量處理 NAS 數據")
        print("  2. 診斷數據質量")
        print("  3. 訓練模型")
        print("  4. 評估模型")
        print("  5. 完整流程")
        print("  0. 退出")
        print("="*80)
        
        choice = input("\n請選擇: ").strip()
        
        if choice == '0':
            break
            
        elif choice == '1':
            nas_root = input(r"NAS 根目錄 [\\10.1.1.3\bio3\PURE_dataset]: ").strip() or r"\\10.1.1.3\bio3\PURE_dataset"
            local_output = input(r"本地輸出目錄 [D:\rppg\motion_reconstruction\rppg_output]: ").strip() or r"D:\rppg\motion_reconstruction\rppg_output"
            batch_process_nas_data(nas_root, local_output)
            
        elif choice == '2':
            data_dir = input(r"數據目錄 [D:\rppg\motion_reconstruction\rppg_output]: ").strip() or r"D:\rppg\motion_reconstruction\rppg_output"
            diagnose_all_subjects(data_dir)
            
        elif choice == '3':
            data_dir = input(r"數據目錄 [D:\rppg\motion_reconstruction\rppg_output]: ").strip() or r"D:\rppg\motion_reconstruction\rppg_output"
            output_dir = input(r"輸出目錄 [D:\rppg\motion_reconstruction\projection_models]: ").strip() or r"D:\rppg\motion_reconstruction\projection_models"
            epochs = int(input("訓練輪數 [50]: ").strip() or 50)
            train_projection_model(data_dir, output_dir, epochs=epochs)
            
        elif choice == '4':
            model_path = input(r"模型路徑 [D:\rppg\motion_reconstruction\projection_models\best_projection_model.pth]: ").strip() or r"D:\rppg\motion_reconstruction\projection_models\best_projection_model.pth"
            data_dir = input(r"數據目錄 [D:\rppg\motion_reconstruction\rppg_output]: ").strip() or r"D:\rppg\motion_reconstruction\rppg_output"
            output_dir = input(r"輸出目錄 [.\evaluation_results]: ").strip() or r".\evaluation_results"
            evaluate_model(model_path, data_dir, output_dir)
            
        elif choice == '5':
            # 完整流程省略，邏輯同上
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['batch', 'diagnose', 'train', 'evaluate', 'full'])
    # ... 其他參數省略，預設互動模式 ...
    args = parser.parse_args()
    
    if args.mode is None:
        interactive_mode()
    else:
        # 命令列模式實作 (略，結構與 interactive 類似)
        pass

if __name__ == "__main__":
    main()