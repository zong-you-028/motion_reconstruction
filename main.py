#!/usr/bin/env python3
"""
Learnable Projection POS - 主程式
One-Click Execution for Learnable Projection Matrix POS

這個版本專門用於訓練和使用可學習投影矩陣的 POS 演算法
"""

import os
import sys
import argparse
from pathlib import Path

def print_banner():
    """顯示歡迎畫面"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║      Learnable Projection Matrix POS (LP-POS)            ║
    ║         可訓練投影矩陣 POS - 一鍵執行                      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    
    訓練整個 2×3 投影矩陣 P(t)，而不只是 alpha 參數
    """
    print(banner)

def check_dependencies():
    """檢查必要的依賴套件"""
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'cv2': 'OpenCV (opencv-python)',
        'mediapipe': 'MediaPipe',
        'matplotlib': 'Matplotlib'
    }
    
    missing_packages = []
    
    print("正在檢查依賴套件...")
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - 未安裝")
            missing_packages.append(name)
    
    if missing_packages:
        print("\n⚠️  缺少以下套件，請先安裝：")
        print(f"pip install {' '.join([p.split()[0].lower().replace('opencv', 'opencv-python') for p in missing_packages])}")
        return False
    
    print("\n✓ 所有依賴套件已安裝\n")
    return True

def show_menu():
    """顯示主選單"""
    menu = """
    ┌─────────────────────────────────────────────────────────┐
    │  請選擇要執行的功能：                                      │
    ├─────────────────────────────────────────────────────────┤
    │  1. 快速測試 (使用合成數據訓練)                           │
    │  2. 訓練模型 (使用實際數據)                               │
    │  3. 評估模型                                             │
    │  4. 提取 ROI (從影片幀)                                  │
    │  5. 使用訓練好的模型推論                                  │
    │  6. 比較三種模型架構                                      │
    │  0. 退出                                                 │
    └─────────────────────────────────────────────────────────┘
    """
    print(menu)

def quick_test():
    """選項 1: 快速測試"""
    print("\n" + "="*60)
    print("選項 1: 快速測試 (使用合成數據)")
    print("="*60)
    
    print("\n這將使用合成數據快速訓練模型")
    print("預計時間：5-10 分鐘\n")
    
    confirm = input("確認開始？(y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消")
        return
    
    try:
        print("\n正在生成測試數據...")
        from demo_usage import generate_synthetic_rppg_data
        import numpy as np
        
        # 生成測試數據
        rgb_traces = []
        ppg_signals = []
        
        for i in range(20):
            print(f"  生成受試者 {i+1}/20...", end='\r')
            r, g, b, ppg, _ = generate_synthetic_rppg_data(
                duration=30, fs=84, 
                hr=np.random.uniform(60, 100),
                motion_strength=np.random.uniform(0.2, 0.6)
            )
            rgb_traces.append((r, g, b))
            ppg_signals.append(ppg)
        
        print("\n✓ 數據生成完成！")
        
        # 訓練
        print("\n開始訓練...")
        from train_projection_pos import train_projection_model, RPPGDatasetForProjection
        from learnable_projection_pos import ConstrainedProjectionPredictor
        from torch.utils.data import DataLoader
        import torch
        
        # 分割數據
        split_idx = int(0.8 * len(rgb_traces))
        train_rgb = rgb_traces[:split_idx]
        train_ppg = ppg_signals[:split_idx]
        val_rgb = rgb_traces[split_idx:]
        val_ppg = ppg_signals[split_idx:]
        
        # 創建數據集
        train_dataset = RPPGDatasetForProjection(
            train_rgb, train_ppg, window_length=128, stride=32, mode='feature'
        )
        val_dataset = RPPGDatasetForProjection(
            val_rgb, val_ppg, window_length=128, stride=64, mode='feature'
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 創建模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n使用設備: {device}")
        
        model = ConstrainedProjectionPredictor(
            input_dim=10, hidden_dim=64, use_residual=True
        )
        print(f"模型參數量: {sum(p.numel() for p in model.parameters())}")
        
        # 訓練
        trained_model, train_losses, val_losses = train_projection_model(
            model, train_loader, val_loader,
            num_epochs=30,
            learning_rate=0.001,
            device=device,
            save_dir='./projection_models',
            model_type='feature'
        )
        
        print("\n" + "="*60)
        print("✓ 訓練完成！")
        print("="*60)
        print(f"\n結果保存在: ./projection_models/")
        print(f"  - best_projection_model.pth")
        print(f"  - projection_training_curves.png")
        print(f"\n最佳驗證損失: {min(val_losses):.4f}")
        
    except Exception as e:
        print(f"\n✗ 錯誤：{e}")
        import traceback
        traceback.print_exc()

def train_with_real_data():
    """選項 2: 使用實際數據訓練"""
    print("\n" + "="*60)
    print("選項 2: 訓練模型 (使用實際數據)")
    print("="*60)
    
    print("\n此功能需要已經提取好的 RGB traces 和 PPG 數據")
    print("數據格式要求：")
    print("  - RGB traces: CSV 檔案 (frame, R_avg, G_avg, B_avg, success)")
    print("  - PPG: .mat 或 .csv 檔案")
    
    data_dir = input("\n請輸入數據目錄路徑: ").strip()
    if not data_dir or not os.path.exists(data_dir):
        print("\n✗ 錯誤：目錄不存在")
        return
    
    output_dir = input("請輸入輸出目錄路徑 [預設: ./projection_models]: ").strip()
    if not output_dir:
        output_dir = './projection_models'
    
    print("\n選擇模型類型：")
    print("  1. ConstrainedProjectionPredictor (推薦，Residual 模式)")
    print("  2. ProjectionMatrixPredictor (基礎)")
    print("  3. TemporalProjectionPredictor (時序，LSTM)")
    
    model_choice = input("請選擇 [預設: 1]: ").strip()
    if not model_choice:
        model_choice = '1'
    
    epochs = input("訓練輪數 [預設: 50]: ").strip()
    num_epochs = int(epochs) if epochs else 50
    
    print("\n準備訓練...")
    print(f"  數據目錄: {data_dir}")
    print(f"  輸出目錄: {output_dir}")
    print(f"  訓練輪數: {num_epochs}")
    
    confirm = input("\n確認執行？(y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消")
        return
    
    try:
        from data_loader import DataLoader
        from train_projection_pos import train_projection_model, RPPGDatasetForProjection
        from learnable_projection_pos import (
            ProjectionMatrixPredictor,
            ConstrainedProjectionPredictor,
            TemporalProjectionPredictor
        )
        from torch.utils.data import DataLoader as TorchDataLoader
        import torch
        
        # 載入數據
        print("\n正在載入數據...")
        
        # 掃描數據目錄
        data_path = Path(data_dir)
        rgb_traces = []
        ppg_signals = []
        
        # 收集所有受試者的檔案路徑
        subject_list = []
        
        # 檢查子目錄
        for subject_dir in data_path.iterdir():
            if subject_dir.is_dir():
                csv_files = list(subject_dir.glob('*rgb_traces.csv'))
                ppg_files = list(subject_dir.glob('ppg.*')) + list(subject_dir.glob('PPG*.csv'))
                
                if csv_files and ppg_files:
                    subject_list.append({
                        'subject_id': subject_dir.name,
                        'rgb_csv_path': str(csv_files[0]),
                        'ppg_path': str(ppg_files[0]),
                        'rgb_start_frame': 0
                    })
        
        # 檢查根目錄（單個受試者情況）
        csv_files_root = list(data_path.glob('*rgb_traces.csv'))
        ppg_files_root = list(data_path.glob('ppg.*')) + list(data_path.glob('PPG*.csv'))
        
        if csv_files_root and ppg_files_root:
            subject_list.append({
                'subject_id': 'single_subject',
                'rgb_csv_path': str(csv_files_root[0]),
                'ppg_path': str(ppg_files_root[0]),
                'rgb_start_frame': 0
            })
        
        if len(subject_list) == 0:
            print("\n✗ 錯誤：未找到有效的數據")
            print("\n請確認數據結構：")
            print("  方式 1（多受試者）：")
            print("    data_dir/")
            print("      subject1/")
            print("        subject1_rgb_traces.csv")
            print("        ppg.csv")
            print("      subject2/")
            print("        subject2_rgb_traces.csv")
            print("        ppg.csv")
            print("\n  方式 2（單受試者）：")
            print("    data_dir/")
            print("      xxx_rgb_traces.csv")
            print("      ppg.csv")
            return
        
        # 使用 DataLoader 載入所有數據
        loader = DataLoader(data_dir=str(data_path), fs=84)
        all_data = loader.load_multiple_subjects(subject_list)
        
        if len(all_data) == 0:
            print("\n✗ 錯誤：未能載入任何數據")
            return
        
        # 轉換為訓練格式
        for data in all_data:
            r = data['r']
            g = data['g']
            b = data['b']
            ppg = data['ppg']
            
            rgb_traces.append((r, g, b))
            ppg_signals.append(ppg)
        
        if len(rgb_traces) == 0:
            print("\n✗ 錯誤：未找到有效的數據")
            return
        
        print(f"\n✓ 載入 {len(rgb_traces)} 個受試者的數據")
        
        # 分割數據
        split_idx = int(0.8 * len(rgb_traces))
        train_rgb = rgb_traces[:split_idx]
        train_ppg = ppg_signals[:split_idx]
        val_rgb = rgb_traces[split_idx:]
        val_ppg = ppg_signals[split_idx:]
        
        print(f"  訓練集: {len(train_rgb)} 個受試者")
        print(f"  驗證集: {len(val_rgb)} 個受試者")
        
        # 創建模型
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n使用設備: {device}")
        
        if model_choice == '1':
            model = ConstrainedProjectionPredictor(input_dim=10, hidden_dim=64, use_residual=True)
            model_type = 'feature'
            dataset_mode = 'feature'
            print("使用模型: ConstrainedProjectionPredictor (Residual)")
        elif model_choice == '2':
            model = ProjectionMatrixPredictor(input_dim=10, hidden_dim=64)
            model_type = 'feature'
            dataset_mode = 'feature'
            print("使用模型: ProjectionMatrixPredictor")
        else:
            model = TemporalProjectionPredictor(window_size=128, hidden_dim=64)
            model_type = 'sequence'
            dataset_mode = 'sequence'
            print("使用模型: TemporalProjectionPredictor (LSTM)")
        
        print(f"模型參數量: {sum(p.numel() for p in model.parameters())}")
        
        # 創建數據集
        print("\n創建數據集...")
        train_dataset = RPPGDatasetForProjection(
            train_rgb, train_ppg, window_length=128, stride=32, mode=dataset_mode
        )
        val_dataset = RPPGDatasetForProjection(
            val_rgb, val_ppg, window_length=128, stride=64, mode=dataset_mode
        )
        
        train_loader = TorchDataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=16, shuffle=False)
        
        print(f"  訓練樣本: {len(train_dataset)}")
        print(f"  驗證樣本: {len(val_dataset)}")
        
        # 訓練
        print("\n開始訓練...")
        trained_model, train_losses, val_losses = train_projection_model(
            model, train_loader, val_loader,
            num_epochs=num_epochs,
            learning_rate=0.001,
            device=device,
            save_dir=output_dir,
            model_type=model_type
        )
        
        print("\n" + "="*60)
        print("✓ 訓練完成！")
        print("="*60)
        print(f"\n結果保存在: {output_dir}/")
        print(f"最佳驗證損失: {min(val_losses):.4f}")
        
    except Exception as e:
        print(f"\n✗ 錯誤：{e}")
        import traceback
        traceback.print_exc()

def evaluate_model():
    """選項 3: 評估模型"""
    print("\n" + "="*60)
    print("選項 3: 評估模型")
    print("="*60)
    
    model_path = input("\n請輸入模型檔案路徑 (.pth): ").strip()
    if not model_path or not os.path.exists(model_path):
        print("\n✗ 錯誤：模型檔案不存在")
        return
    
    data_dir = input("請輸入測試數據目錄: ").strip()
    if not data_dir or not os.path.exists(data_dir):
        print("\n✗ 錯誤：目錄不存在")
        return
    
    print("\n功能開發中...")
    print("目前請使用 evaluate_adaptive_pos.py 進行評估")

def extract_roi():
    """選項 4: 提取 ROI"""
    print("\n" + "="*60)
    print("選項 4: 提取 ROI")
    print("="*60)
    
    frames_folder = input("\n請輸入影片幀所在目錄 (可以是 NAS 路徑): ").strip()
    if not frames_folder or not os.path.exists(frames_folder):
        print("\n✗ 錯誤：目錄不存在")
        return
    
    subject_id = input("請輸入受試者 ID: ").strip()
    if not subject_id:
        print("\n✗ 錯誤：需要提供受試者 ID")
        return
    
    output_dir = input("請輸入輸出目錄 (本地路徑，預設: ./FaceMesh_Output): ").strip()
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), 'FaceMesh_Output')
    
    print(f"\n📁 讀取來源: {frames_folder}")
    print(f"💾 輸出目錄: {output_dir}")
    
    try:
        from facemesh_roi_cheeks_only import process_roi_extraction
        import pandas as pd
        
        print(f"\n正在處理 {subject_id}...")
        results = process_roi_extraction(frames_folder, subject_id, output_dir)
        
        # 保存 CSV
        csv_path = os.path.join(output_dir, f'{subject_id}_rgb_traces.csv')
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ ROI 提取完成！")
        print(f"結果已輸出（共 {len(results)} 幀）")
        print(f"CSV 檔案: {csv_path}")
        
    except Exception as e:
        print(f"\n✗ 錯誤：{e}")
        import traceback
        traceback.print_exc()

def inference_with_model():
    """選項 5: 使用訓練好的模型推論"""
    print("\n" + "="*60)
    print("選項 5: 模型推論")
    print("="*60)
    
    model_path = input("\n請輸入模型檔案路徑 (.pth): ").strip()
    if not model_path or not os.path.exists(model_path):
        print("\n✗ 錯誤：模型檔案不存在")
        return
    
    csv_path = input("請輸入 RGB traces CSV 檔案路徑: ").strip()
    if not csv_path or not os.path.exists(csv_path):
        print("\n✗ 錯誤：CSV 檔案不存在")
        return
    
    try:
        import torch
        import pandas as pd
        import numpy as np
        from learnable_projection_pos import (
            ConstrainedProjectionPredictor,
            LearnableProjectionPOS
        )
        from scipy.stats import pearsonr
        
        # 載入模型
        print("\n正在載入模型...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = ConstrainedProjectionPredictor(input_dim=10, hidden_dim=64, use_residual=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ 模型載入完成")
        
        # 載入數據
        print("\n正在載入 RGB 數據...")
        df = pd.read_csv(csv_path)
        df = df[df['success'] == 1]
        
        r_buf = df['R_avg'].values
        g_buf = df['G_avg'].values
        b_buf = df['B_avg'].values
        print(f"✓ 數據載入完成（{len(r_buf)} 幀）")
        
        # 處理
        print("\n正在處理...")
        pos = LearnableProjectionPOS(window_length=128, fs=84)
        
        # 標準 POS
        rppg_standard = pos.process_standard(r_buf, g_buf, b_buf)
        
        # 可學習投影矩陣 POS
        rppg_learnable, P_history = pos.process_learnable(
            r_buf, g_buf, b_buf, model, use_features=True
        )
        
        # 計算心率
        from learnable_projection_pos import calculate_hr_from_rppg
        hr_std = calculate_hr_from_rppg(rppg_standard[128:], fs=84)
        hr_learn = calculate_hr_from_rppg(rppg_learnable[128:], fs=84)
        
        # 顯示結果
        print("\n" + "="*60)
        print("處理完成！")
        print("="*60)
        
        print(f"\n標準 POS:")
        print(f"  心率: {hr_std:.1f} bpm")
        
        print(f"\n可學習投影矩陣 POS:")
        print(f"  心率: {hr_learn:.1f} bpm")
        
        print(f"\n投影矩陣變化:")
        print(f"  標準 POS: [[0, 1, -1], [-2, 1, 1]]")
        print(f"  學習到的 P(t=0):\n    {P_history[0]}")
        P_standard = np.array([[0, 1, -1], [-2, 1, 1]])
        deviation = np.mean(np.abs(P_history - P_standard))
        print(f"  平均偏離標準 POS: {deviation:.3f}")
        
        # 可視化
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle('投影矩陣隨時間變化', fontsize=16)
            
            for i in range(2):
                for j in range(3):
                    ax = axes[i, j]
                    ax.plot(P_history[:, i, j], label='Learned', linewidth=2)
                    ax.axhline(P_standard[i, j], color='r', linestyle='--', 
                             label='Standard', linewidth=2)
                    ax.set_title(f'P[{i},{j}]')
                    ax.set_xlabel('Time (frames)')
                    ax.set_ylabel('Value')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_path = 'projection_matrix_evolution.png'
            plt.savefig(output_path, dpi=150)
            print(f"\n可視化圖已保存: {output_path}")
        except:
            pass
        
    except Exception as e:
        print(f"\n✗ 錯誤：{e}")
        import traceback
        traceback.print_exc()

def compare_models():
    """選項 6: 比較三種模型架構"""
    print("\n" + "="*60)
    print("選項 6: 比較三種模型架構")
    print("="*60)
    
    print("\n將訓練和比較三種模型：")
    print("  1. ProjectionMatrixPredictor (基礎)")
    print("  2. ConstrainedProjectionPredictor (Residual)")
    print("  3. TemporalProjectionPredictor (LSTM)")
    
    print("\n預計時間：15-30 分鐘")
    
    confirm = input("\n確認開始？(y/n) [y]: ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消")
        return
    
    print("\n功能開發中...")
    print("目前請分別運行選項 1 三次，每次選擇不同的模型")

def main():
    """主程式"""
    # 顯示歡迎畫面
    print_banner()
    
    # 檢查依賴
    if not check_dependencies():
        sys.exit(1)
    
    # 主循環
    while True:
        show_menu()
        choice = input("請選擇 (0-6): ").strip()
        
        if choice == '0':
            print("\n再見！👋")
            break
        elif choice == '1':
            quick_test()
        elif choice == '2':
            train_with_real_data()
        elif choice == '3':
            evaluate_model()
        elif choice == '4':
            extract_roi()
        elif choice == '5':
            inference_with_model()
        elif choice == '6':
            compare_models()
        else:
            print("\n✗ 無效的選擇，請重新輸入")
        
        # 等待用戶
        input("\n按 Enter 繼續...")
        print("\n" * 2)

if __name__ == "__main__":
    # 支援命令列參數（快速模式）
    parser = argparse.ArgumentParser(description='Learnable Projection POS - 一鍵執行')
    parser.add_argument('--quick-test', action='store_true', help='直接運行快速測試')
    parser.add_argument('--train', type=str, help='訓練模型（指定數據目錄）')
    parser.add_argument('--inference', nargs=2, metavar=('MODEL', 'CSV'), 
                       help='推論模式（模型路徑 CSV路徑）')
    
    args = parser.parse_args()
    
    if args.quick_test:
        # 快速模式
        print_banner()
        quick_test()
    elif args.train:
        # 訓練模式
        print_banner()
        # TODO: 實現快速訓練
        print("命令列快速模式開發中...")
    elif args.inference:
        # 推論模式
        print_banner()
        # TODO: 實現快速推論
        print("命令列推論模式開發中...")
    else:
        # 互動式選單模式
        main()