"""
PURE 資料集分割腳本
按受試者分割：某些受試者的所有場景作為 train，其他受試者的所有場景作為 test
"""

import os
import shutil
from pathlib import Path
import random

def split_pure_dataset(source_dir, output_base_dir, test_ratio=0.2, seed=42):
    """
    分割 PURE 資料集（按受試者）
    
    策略：
    - 將某些受試者的**所有場景**分配給 train
    - 將其他受試者的**所有場景**分配給 test
    - 這樣可以測試模型在新受試者上的泛化能力
    
    Args:
        source_dir: 原始資料目錄（D:\rppg\motion_reconstruction\rppg_output）
        output_base_dir: 輸出基礎目錄
        test_ratio: 測試集受試者比例
        seed: 隨機種子
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_base_dir)
    
    # 創建輸出目錄
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("PURE 資料集分割（按受試者）")
    print("="*80)
    print(f"\n來源目錄: {source_dir}")
    print(f"輸出目錄: {output_base_dir}")
    print(f"測試集受試者比例: {test_ratio*100:.0f}%")
    print(f"隨機種子: {seed}")
    
    # 掃描所有資料夾
    all_folders = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    
    print(f"\n找到 {len(all_folders)} 個資料夾:")
    print(all_folders)
    
    # 按照受試者 ID 分組（01, 02, 03, 04, 05）
    subject_groups = {}
    for folder in all_folders:
        # 提取受試者 ID (01, 02, 03, 04, 05)
        subject_id = folder.split('-')[0]
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append(folder)
    
    print(f"\n受試者分組:")
    for sid, folders in sorted(subject_groups.items()):
        print(f"  受試者 {sid}: {folders} ({len(folders)} 個場景)")
    
    # 取得所有受試者 ID
    subject_ids = sorted(subject_groups.keys())
    print(f"\n總共 {len(subject_ids)} 個受試者: {subject_ids}")
    
    # 隨機打亂受試者順序
    shuffled_ids = subject_ids.copy()
    random.shuffle(shuffled_ids)
    
    # 計算測試集受試者數量
    n_test_subjects = max(1, int(len(shuffled_ids) * test_ratio))
    n_train_subjects = len(shuffled_ids) - n_test_subjects
    
    # 分配受試者
    test_subject_ids = shuffled_ids[:n_test_subjects]
    train_subject_ids = shuffled_ids[n_test_subjects:]
    
    print(f"\n分配結果:")
    print(f"  Train 受試者 ({n_train_subjects} 個): {sorted(train_subject_ids)}")
    print(f"  Test 受試者  ({n_test_subjects} 個): {sorted(test_subject_ids)}")
    
    # 收集所有 train 和 test 資料夾
    train_folders = []
    test_folders = []
    
    for sid in train_subject_ids:
        train_folders.extend(subject_groups[sid])
    
    for sid in test_subject_ids:
        test_folders.extend(subject_groups[sid])
    
    print(f"\n資料夾分配:")
    print(f"  Train 資料夾 ({len(train_folders)} 個): {sorted(train_folders)}")
    print(f"  Test 資料夾  ({len(test_folders)} 個): {sorted(test_folders)}")
    
    # 驗證：確保每個場景都有資料在兩個 set
    print(f"\n驗證分割...")
    
    # 提取場景編號
    train_scenes = set([f.split('-')[1] for f in train_folders])
    test_scenes = set([f.split('-')[1] for f in test_folders])
    
    print(f"  Train 包含場景: {sorted(train_scenes)}")
    print(f"  Test 包含場景: {sorted(test_scenes)}")
    
    all_scenes_expected = set(['01', '02', '03', '04', '05', '06'])
    missing_in_train = all_scenes_expected - train_scenes
    missing_in_test = all_scenes_expected - test_scenes
    
    if missing_in_train:
        print(f"  ⚠️  警告: Train 缺少場景 {missing_in_train}")
    if missing_in_test:
        print(f"  ⚠️  警告: Test 缺少場景 {missing_in_test}")
    
    if not missing_in_train and not missing_in_test:
        print(f"  ✓ 兩個 set 都包含所有場景")
    
    # 複製檔案
    print(f"\n開始複製檔案...")
    
    copied_train = 0
    for folder in train_folders:
        src = source_path / folder
        dst = train_dir / folder
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  ✓ Train: {folder}")
            copied_train += 1
    
    copied_test = 0
    for folder in test_folders:
        src = source_path / folder
        dst = test_dir / folder
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  ✓ Test: {folder}")
            copied_test += 1
    
    # 生成摘要
    print("\n" + "="*80)
    print("分割完成")
    print("="*80)
    
    print(f"\nTrain Set:")
    print(f"  受試者: {sorted(train_subject_ids)} ({n_train_subjects} 個)")
    print(f"  資料夾: {copied_train} 個")
    print(f"  場景覆蓋: {sorted(train_scenes)}")
    
    print(f"\nTest Set:")
    print(f"  受試者: {sorted(test_subject_ids)} ({n_test_subjects} 個)")
    print(f"  資料夾: {copied_test} 個")
    print(f"  場景覆蓋: {sorted(test_scenes)}")
    
    # 保存分割記錄
    split_info_path = output_path / "split_info.txt"
    with open(split_info_path, 'w', encoding='utf-8') as f:
        f.write("PURE 資料集分割記錄（按受試者分割）\n")
        f.write("="*80 + "\n\n")
        f.write(f"日期: {__import__('datetime').datetime.now()}\n")
        f.write(f"來源: {source_dir}\n")
        f.write(f"測試集受試者比例: {test_ratio*100:.0f}%\n")
        f.write(f"隨機種子: {seed}\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Train Set ({n_train_subjects} 個受試者，{copied_train} 個資料夾)\n")
        f.write("="*80 + "\n")
        f.write(f"受試者: {sorted(train_subject_ids)}\n\n")
        f.write("資料夾:\n")
        for folder in sorted(train_folders):
            f.write(f"  {folder}\n")
        f.write(f"\n場景覆蓋: {sorted(train_scenes)}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Test Set ({n_test_subjects} 個受試者，{copied_test} 個資料夾)\n")
        f.write("="*80 + "\n")
        f.write(f"受試者: {sorted(test_subject_ids)}\n\n")
        f.write("資料夾:\n")
        for folder in sorted(test_folders):
            f.write(f"  {folder}\n")
        f.write(f"\n場景覆蓋: {sorted(test_scenes)}\n")
    
    print(f"\n分割記錄已保存: {split_info_path}")
    print("="*80)
    
    return train_folders, test_folders


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PURE 資料集分割（按受試者）')
    parser.add_argument('--source', type=str, 
                       default=r'D:\rppg\motion_reconstruction\rppg_output',
                       help='原始資料目錄')
    parser.add_argument('--output', type=str, 
                       default=r'D:\rppg\motion_reconstruction\rppg_split',
                       help='輸出基礎目錄')
    parser.add_argument('--test-ratio', type=float, 
                       default=0.2,
                       help='測試集受試者比例 (0.0-1.0)')
    parser.add_argument('--seed', type=int, 
                       default=42,
                       help='隨機種子')
    
    args = parser.parse_args()
    
    # 執行分割
    train_folders, test_folders = split_pure_dataset(
        source_dir=args.source,
        output_base_dir=args.output,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"\n✅ 完成！")
    print(f"   Train 目錄: {args.output}\\train ({len(train_folders)} 個資料夾)")
    print(f"   Test 目錄: {args.output}\\test ({len(test_folders)} 個資料夾)")


if __name__ == "__main__":
    main()