"""
批量處理 NAS 上的 PURE 數據集
從 NAS 提取 ROI + 複製 PPG → 保存到本地

數據結構：
NAS: \\10.1.1.3\bio3\PURE_dataset\
├── 01-01\
│   ├── cam0\           # BMP 影片幀
│   │   ├── 0001.bmp
│   │   └── ...
│   └── PPG_CMS50E_30fps.csv
├── 01-02\
│   ├── cam0\
│   └── PPG_CMS50E_30fps.csv
└── ...
"""

import subprocess
import shutil
from pathlib import Path
import sys

# ==================== 配置區 ====================
# NAS 根目錄
NAS_ROOT = r"\\10.1.1.3\bio3\PURE_dataset"

# 本地輸出目錄
LOCAL_OUTPUT = r"D:\rppg_output"

# 要處理的受試者列表
# 留空 [] = 自動處理 NAS_ROOT 下的所有資料夾 ⭐
SUBJECTS = []

# 子資料夾名稱（影片幀所在的資料夾）
FRAMES_FOLDER = "cam0"  # 如果你的影片在其他資料夾，改成對應名稱

# PPG 檔案名稱
PPG_FILENAME = "PPG_CMS50E_30fps.csv"
# ================================================


def scan_subjects(nas_root):
    """
    掃描 NAS 目錄，找到所有受試者
    
    Returns:
        list: 受試者 ID 列表
    """
    nas_path = Path(nas_root)
    
    if not nas_path.exists():
        print(f"✗ 錯誤：NAS 目錄不存在: {nas_root}")
        return []
    
    subjects = []
    
    print(f"正在掃描 NAS 目錄: {nas_root}")
    print("="*60)
    
    for item in nas_path.iterdir():
        if item.is_dir():
            # 檢查是否有 cam0 資料夾和 PPG 檔案
            frames_dir = item / FRAMES_FOLDER
            ppg_file = item / PPG_FILENAME
            
            if frames_dir.exists() and ppg_file.exists():
                subjects.append(item.name)
                print(f"  ✓ {item.name}")
            else:
                print(f"  ✗ {item.name} (缺少檔案)")
                if not frames_dir.exists():
                    print(f"      缺少: {FRAMES_FOLDER}")
                if not ppg_file.exists():
                    print(f"      缺少: {PPG_FILENAME}")
    
    print("="*60)
    print(f"找到 {len(subjects)} 個有效受試者\n")
    
    return sorted(subjects)


def process_subject(subject_id, nas_root, local_output, frames_folder, ppg_filename):
    """
    處理單個受試者：提取 ROI + 複製 PPG
    
    Args:
        subject_id: 受試者 ID (例如 "01-01")
        nas_root: NAS 根目錄
        local_output: 本地輸出目錄
        frames_folder: 影片幀所在的子資料夾名稱
        ppg_filename: PPG 檔案名稱
    
    Returns:
        bool: 是否成功
    """
    print("\n" + "="*60)
    print(f"處理受試者: {subject_id}")
    print("="*60)
    
    # 路徑設定
    nas_subject_dir = Path(nas_root) / subject_id
    nas_frames_dir = nas_subject_dir / frames_folder
    nas_ppg_file = nas_subject_dir / ppg_filename
    
    local_subject_dir = Path(local_output) / subject_id
    local_subject_dir.mkdir(parents=True, exist_ok=True)
    
    # 步驟 1: 提取 ROI
    print(f"\n步驟 1: 提取 ROI from NAS")
    print(f"  來源: {nas_frames_dir}")
    print(f"  輸出: {local_output}")
    
    try:
        # 調用 facemesh_roi_cheeks_only.py
        result = subprocess.run([
            sys.executable,  # 當前 Python 解釋器
            "facemesh_roi_cheeks_only.py",
            str(nas_frames_dir),
            subject_id,
            str(local_output)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  ✓ ROI 提取成功")
            
            # facemesh 會輸出到 stdout，需要解析並保存為 CSV
            output_lines = result.stdout.strip().split('\n')
            
            # 過濾掉非數據行（可能有 print 訊息）
            data_lines = []
            for line in output_lines:
                if ',' in line and not line.startswith('frame'):
                    try:
                        # 驗證是否為有效數據行
                        parts = line.split(',')
                        if len(parts) == 5:
                            float(parts[0])  # frame
                            float(parts[1])  # R_avg
                            data_lines.append(line)
                    except:
                        continue
            
            if data_lines:
                # 保存 CSV
                csv_path = local_subject_dir / f"{subject_id}_rgb_traces.csv"
                with open(csv_path, 'w') as f:
                    f.write("frame,R_avg,G_avg,B_avg,success\n")
                    f.write('\n'.join(data_lines))
                print(f"  ✓ 保存 CSV: {csv_path.name} ({len(data_lines)} 幀)")
            else:
                print(f"  ✗ 未獲取到有效數據")
                return False
        else:
            print(f"  ✗ ROI 提取失敗")
            if result.stderr:
                print(f"  錯誤: {result.stderr[:200]}")  # 只顯示前 200 字元
            return False
            
    except Exception as e:
        print(f"  ✗ ROI 提取異常: {e}")
        return False
    
    # 步驟 2: 複製 PPG
    print(f"\n步驟 2: 複製 PPG from NAS")
    print(f"  來源: {nas_ppg_file}")
    
    try:
        local_ppg_file = local_subject_dir / "ppg.csv"
        
        if nas_ppg_file.exists():
            shutil.copy(nas_ppg_file, local_ppg_file)
            print(f"  ✓ PPG 複製成功")
            print(f"  目標: {local_ppg_file}")
        else:
            print(f"  ✗ PPG 檔案不存在: {nas_ppg_file}")
            return False
            
    except Exception as e:
        print(f"  ✗ PPG 複製異常: {e}")
        return False
    
    # 步驟 3: 驗證
    print(f"\n步驟 3: 驗證輸出")
    
    expected_rgb_csv = local_subject_dir / f"{subject_id}_rgb_traces.csv"
    expected_ppg_csv = local_subject_dir / "ppg.csv"
    
    success = True
    
    if expected_rgb_csv.exists():
        print(f"  ✓ RGB CSV: {expected_rgb_csv.name}")
    else:
        print(f"  ✗ RGB CSV 不存在")
        success = False
    
    if expected_ppg_csv.exists():
        print(f"  ✓ PPG CSV: {expected_ppg_csv.name}")
    else:
        print(f"  ✗ PPG CSV 不存在")
        success = False
    
    if success:
        print(f"\n✅ {subject_id} 處理完成")
    else:
        print(f"\n❌ {subject_id} 處理失敗")
    
    return success


def main():
    """主函數"""
    print("="*60)
    print("批量處理 PURE 數據集")
    print("="*60)
    
    print(f"\nNAS 根目錄: {NAS_ROOT}")
    print(f"本地輸出: {LOCAL_OUTPUT}")
    print(f"影片幀資料夾: {FRAMES_FOLDER}")
    print(f"PPG 檔案名稱: {PPG_FILENAME}")
    
    # 確認本地輸出目錄
    local_path = Path(LOCAL_OUTPUT)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # 決定要處理的受試者
    if SUBJECTS:
        # 使用指定的受試者列表
        subjects = SUBJECTS
        print(f"\n使用指定的受試者列表 ({len(subjects)} 個)")
    else:
        # 自動掃描所有受試者
        subjects = scan_subjects(NAS_ROOT)
        
        if not subjects:
            print("\n✗ 未找到任何有效受試者")
            return
    
    # 顯示將要處理的受試者
    print("\n將要處理的受試者:")
    for i, subj in enumerate(subjects, 1):
        print(f"  {i}. {subj}")
    
    # 確認
    print("\n" + "="*60)
    confirm = input(f"確認處理 {len(subjects)} 個受試者？(y/n) [y]: ").strip().lower()
    
    if confirm and confirm != 'y':
        print("已取消")
        return
    
    # 開始批量處理
    print("\n" + "="*60)
    print("開始批量處理...")
    print("="*60)
    
    success_count = 0
    failed_subjects = []
    
    for i, subject_id in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] 處理 {subject_id}...")
        
        try:
            success = process_subject(
                subject_id, 
                NAS_ROOT, 
                LOCAL_OUTPUT, 
                FRAMES_FOLDER, 
                PPG_FILENAME
            )
            
            if success:
                success_count += 1
            else:
                failed_subjects.append(subject_id)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  使用者中斷")
            break
        except Exception as e:
            print(f"\n✗ 處理 {subject_id} 時發生異常: {e}")
            failed_subjects.append(subject_id)
    
    # 總結
    print("\n" + "="*60)
    print("處理完成！")
    print("="*60)
    
    print(f"\n成功: {success_count}/{len(subjects)}")
    
    if failed_subjects:
        print(f"\n失敗的受試者 ({len(failed_subjects)}):")
        for subj in failed_subjects:
            print(f"  - {subj}")
    
    print(f"\n輸出目錄: {LOCAL_OUTPUT}")
    print("\n下一步：訓練模型")
    print("  python main.py")
    print("  選擇 2 (訓練模型)")
    print(f"  輸入數據目錄: {LOCAL_OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 程式異常: {e}")
        import traceback
        traceback.print_exc()