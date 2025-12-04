import os
import numpy as np
import pandas as pd
from scipy import signal, io
import matplotlib.pyplot as plt

class DataLoader:
    """
    Load RGB traces from FaceMesh extraction and PPG ground truth
    """
    def __init__(self, data_dir, fs=84):
        self.data_dir = data_dir
        self.fs = fs
        
    def load_rgb_from_csv(self, csv_path):
        """
        Load RGB traces from FaceMesh output CSV
        Expected format: frame, R_avg, G_avg, B_avg, success
        """
        df = pd.read_csv(csv_path)
        
        # 檢查必要的欄位
        required_cols = ['R_avg', 'G_avg', 'B_avg']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter successful frames (如果有 success 欄位)
        if 'success' in df.columns:
            # 嘗試不同的 success 值表示
            if df['success'].dtype == 'object':
                # 可能是字串 'True'/'False' 或 '1'/'0'
                df_filtered = df[df['success'].isin([1, '1', True, 'True', 'true'])]
            else:
                # 數值型
                df_filtered = df[df['success'] == 1]
            
            if len(df_filtered) == 0:
                # 如果沒有 success==1 的，檢查是否所有幀都可用
                print(f"  Warning: No frames with success==1, using all {len(df)} frames")
                df_filtered = df
        else:
            # 沒有 success 欄位，使用所有幀
            print(f"  Note: No 'success' column, using all {len(df)} frames")
            df_filtered = df
        
        # 進一步過濾：移除 RGB 全為 0 的幀（通常是偵測失敗）
        valid_mask = (df_filtered['R_avg'] > 0) | (df_filtered['G_avg'] > 0) | (df_filtered['B_avg'] > 0)
        df_filtered = df_filtered[valid_mask]
        
        if len(df_filtered) == 0:
            raise ValueError(f"No valid frames in {csv_path} (all RGB values are 0)")
        
        r_trace = df_filtered['R_avg'].values
        g_trace = df_filtered['G_avg'].values
        b_trace = df_filtered['B_avg'].values
        
        return r_trace, g_trace, b_trace
    
    def load_ppg_from_mat(self, mat_path, signal_name='ppg'):
        """
        Load PPG signal from MATLAB .mat file
        
        Args:
            mat_path: path to .mat file
            signal_name: name of the variable in .mat file
        """
        mat_data = io.loadmat(mat_path)
        
        if signal_name in mat_data:
            ppg = mat_data[signal_name].flatten()
        else:
            # Try to find the signal automatically
            possible_names = ['ppg', 'PPG', 'signal', 'data', 'BVP']
            for name in possible_names:
                if name in mat_data:
                    ppg = mat_data[name].flatten()
                    print(f"Found PPG signal with name: {name}")
                    break
            else:
                raise ValueError(f"Could not find PPG signal in {mat_path}. Available keys: {mat_data.keys()}")
        
        return ppg
    
    def load_ppg_from_csv(self, csv_path, ppg_column='PPG'):
        """
        Load PPG signal from CSV file
        支援多種格式：
        - 無 header（只有數據）
        - 單列（只有 PPG 值）
        - 多列（有時間戳、PPG、SpO2 等）
        """
        # 先嘗試正常載入（有 header）
        df = pd.read_csv(csv_path)
        
        # 檢查第一行是否是數據（判斷是否缺少 header）
        first_col_name = df.columns[0]
        
        # 如果第一個欄位名稱看起來像數字，表示沒有 header
        try:
            # 嘗試將欄位名稱轉換為數字
            float(first_col_name)
            # 如果成功，表示這是數據，需要重新載入
            print(f"  Note: CSV has no header, loading as pure data")
            df = pd.read_csv(csv_path, header=None)
            
            # 使用第一列作為 PPG
            if len(df.columns) == 1:
                ppg = df[0].values
            else:
                # 如果有多列，使用第一個數值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                ppg = df[numeric_cols[0]].values
            
            print(f"  Loaded {len(ppg)} PPG samples from column 0")
            return ppg
            
        except (ValueError, TypeError):
            # 欄位名稱不是數字，表示有正常的 header
            pass
        
        # 嘗試找到 PPG 欄位
        if ppg_column not in df.columns:
            # 常見的 PPG 欄位名稱
            possible_names = ['PPG', 'ppg', 'Ppg', 'BVP', 'bvp', 'signal', 'Signal', 'value', 'Value', 'pleth', 'Pleth']
            
            for name in possible_names:
                if name in df.columns:
                    ppg_column = name
                    print(f"  Found PPG column: '{name}'")
                    break
            else:
                # 如果找不到，使用第一個數值欄位（排除時間戳）
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                # 排除可能的時間或索引欄位
                time_like = ['time', 'Time', 'timestamp', 'Timestamp', 'frame', 'Frame', 'index', 'Index']
                numeric_cols = [col for col in numeric_cols if col not in time_like]
                
                if numeric_cols:
                    ppg_column = numeric_cols[0]
                    print(f"  Using first numeric column as PPG: '{ppg_column}'")
                else:
                    raise ValueError(f"Could not find PPG column in {csv_path}. Available columns: {df.columns.tolist()}")
        
        ppg = df[ppg_column].values
        
        # 驗證 PPG 訊號
        if len(ppg) == 0:
            raise ValueError(f"PPG signal is empty in {csv_path}")
        
        if np.all(ppg == 0):
            print(f"  Warning: PPG signal is all zeros in {csv_path}")
        
        return ppg
    
    def align_rgb_ppg(self, r_trace, g_trace, b_trace, ppg, 
                      rgb_start_frame=300, rgb_fps=84, ppg_fps=84):
        """
        Align RGB traces with PPG signal
        
        Args:
            r_trace, g_trace, b_trace: RGB traces from video
            ppg: PPG signal
            rgb_start_frame: starting frame number of RGB extraction
            rgb_fps: video frame rate
            ppg_fps: PPG sampling rate
        
        Returns:
            aligned RGB traces and PPG
        """
        # Calculate time offset
        rgb_start_time = rgb_start_frame / rgb_fps
        ppg_start_idx = int(rgb_start_time * ppg_fps)
        
        # Length matching
        rgb_length = len(r_trace)
        ppg_length = len(ppg) - ppg_start_idx
        
        if rgb_fps != ppg_fps:
            # Resample if rates differ
            ppg_aligned = signal.resample(ppg[ppg_start_idx:], rgb_length)
        else:
            # Direct alignment
            min_length = min(rgb_length, ppg_length)
            r_trace = r_trace[:min_length]
            g_trace = g_trace[:min_length]
            b_trace = b_trace[:min_length]
            ppg_aligned = ppg[ppg_start_idx:ppg_start_idx + min_length]
        
        return r_trace, g_trace, b_trace, ppg_aligned
    
    def preprocess_signals(self, r_trace, g_trace, b_trace, ppg, 
                          detrend=True, normalize=True):
        """
        Preprocess RGB and PPG signals
        """
        # Detrend
        if detrend:
            r_trace = signal.detrend(r_trace)
            g_trace = signal.detrend(g_trace)
            b_trace = signal.detrend(b_trace)
            ppg = signal.detrend(ppg)
        
        # Normalize
        if normalize:
            r_trace = (r_trace - np.mean(r_trace)) / (np.std(r_trace) + 1e-8)
            g_trace = (g_trace - np.mean(g_trace)) / (np.std(g_trace) + 1e-8)
            b_trace = (b_trace - np.mean(b_trace)) / (np.std(b_trace) + 1e-8)
            ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        
        return r_trace, g_trace, b_trace, ppg
    
    def bandpass_filter(self, signal_data, lowcut=0.7, highcut=4.0, order=4):
        """
        Apply bandpass filter to signal (for HR range 42-240 bpm)
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, signal_data)
        return filtered
    
    def load_subject_data(self, subject_id, rgb_csv_path, ppg_path, 
                         rgb_start_frame=300, visualize=False):
        """
        Load complete data for one subject
        
        Args:
            subject_id: identifier for the subject
            rgb_csv_path: path to FaceMesh output CSV
            ppg_path: path to PPG file (.mat or .csv)
            rgb_start_frame: starting frame of RGB extraction
            visualize: if True, plot signals
        
        Returns:
            dict with aligned and preprocessed data
        """
        print(f"Loading subject: {subject_id}")
        
        # Load RGB
        r_trace, g_trace, b_trace = self.load_rgb_from_csv(rgb_csv_path)
        print(f"  RGB traces loaded: {len(r_trace)} frames")
        
        # Load PPG
        if ppg_path.endswith('.mat'):
            ppg = self.load_ppg_from_mat(ppg_path)
        elif ppg_path.endswith('.csv'):
            ppg = self.load_ppg_from_csv(ppg_path)
        else:
            raise ValueError(f"Unsupported PPG file format: {ppg_path}")
        print(f"  PPG loaded: {len(ppg)} samples")
        
        # Align
        r_trace, g_trace, b_trace, ppg = self.align_rgb_ppg(
            r_trace, g_trace, b_trace, ppg, 
            rgb_start_frame=rgb_start_frame
        )
        print(f"  Aligned length: {len(r_trace)} samples")
        
        # Preprocess
        r_trace, g_trace, b_trace, ppg = self.preprocess_signals(
            r_trace, g_trace, b_trace, ppg
        )
        
        # Visualize if requested
        if visualize:
            self.visualize_signals(subject_id, r_trace, g_trace, b_trace, ppg)
        
        return {
            'subject_id': subject_id,
            'r': r_trace,
            'g': g_trace,
            'b': b_trace,
            'ppg': ppg,
            'fs': self.fs
        }
    
    def load_multiple_subjects(self, subject_list):
        """
        Load data for multiple subjects
        
        Args:
            subject_list: list of dicts, each containing:
                - 'subject_id'
                - 'rgb_csv_path'
                - 'ppg_path'
                - 'rgb_start_frame' (optional, default 300)
        
        Returns:
            list of data dictionaries
        """
        all_data = []
        
        for subject_info in subject_list:
            try:
                data = self.load_subject_data(
                    subject_id=subject_info['subject_id'],
                    rgb_csv_path=subject_info['rgb_csv_path'],
                    ppg_path=subject_info['ppg_path'],
                    rgb_start_frame=subject_info.get('rgb_start_frame', 300)
                )
                all_data.append(data)
            except Exception as e:
                print(f"  Error loading {subject_info['subject_id']}: {e}")
                continue
        
        print(f"\nSuccessfully loaded {len(all_data)} subjects")
        return all_data
    
    def visualize_signals(self, subject_id, r_trace, g_trace, b_trace, ppg):
        """
        Visualize RGB and PPG signals
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # RGB traces
        t = np.arange(len(r_trace)) / self.fs
        axes[0].plot(t, r_trace, 'r-', label='R', alpha=0.7)
        axes[0].plot(t, g_trace, 'g-', label='G', alpha=0.7)
        axes[0].plot(t, b_trace, 'b-', label='B', alpha=0.7)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Normalized Intensity')
        axes[0].set_title(f'{subject_id} - RGB Traces')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # PPG
        axes[1].plot(t, ppg, 'k-', linewidth=1.5)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Normalized Amplitude')
        axes[1].set_title(f'{subject_id} - Ground Truth PPG')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{subject_id}_signals.png', dpi=150)
        plt.close()
        
        print(f"  Visualization saved: {subject_id}_signals.png")


def create_training_dataset_from_files(data_dir, output_file='dataset.npz'):
    """
    Create training dataset from directory structure
    
    Expected structure:
    data_dir/
        subject001/
            rgb_traces.csv
            ppg.mat (or ppg.csv)
        subject002/
            ...
    """
    loader = DataLoader(data_dir, fs=84)
    
    # Find all subjects
    subject_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    subject_list = []
    for subject_dir in subject_dirs:
        subject_path = os.path.join(data_dir, subject_dir)
        
        # Find RGB CSV
        rgb_files = [f for f in os.listdir(subject_path) 
                    if f.endswith('.csv') and 'rgb' in f.lower()]
        if not rgb_files:
            print(f"Warning: No RGB CSV found for {subject_dir}")
            continue
        
        # Find PPG file
        ppg_files = [f for f in os.listdir(subject_path) 
                    if f.endswith(('.mat', '.csv')) and 'ppg' in f.lower()]
        if not ppg_files:
            print(f"Warning: No PPG file found for {subject_dir}")
            continue
        
        subject_list.append({
            'subject_id': subject_dir,
            'rgb_csv_path': os.path.join(subject_path, rgb_files[0]),
            'ppg_path': os.path.join(subject_path, ppg_files[0])
        })
    
    # Load all subjects
    all_data = loader.load_multiple_subjects(subject_list)
    
    # Save as npz
    rgb_traces = [(d['r'], d['g'], d['b']) for d in all_data]
    ppg_signals = [d['ppg'] for d in all_data]
    subject_ids = [d['subject_id'] for d in all_data]
    
    np.savez(output_file, 
             rgb_traces=rgb_traces, 
             ppg_signals=ppg_signals,
             subject_ids=subject_ids,
             fs=84)
    
    print(f"\nDataset saved to {output_file}")
    print(f"Total subjects: {len(all_data)}")
    
    return rgb_traces, ppg_signals


if __name__ == "__main__":
    # Example usage
    print("Data Loader for Adaptive POS")
    print("=" * 60)
    
    # Example 1: Load single subject
    loader = DataLoader(data_dir='./', fs=84)
    
    # Example subject data (replace with your actual paths)
    # subject_data = loader.load_subject_data(
    #     subject_id='subject001',
    #     rgb_csv_path='subject001_rgb_traces.csv',
    #     ppg_path='subject001_ppg.mat',
    #     visualize=True
    # )
    
    # Example 2: Load multiple subjects
    subject_list = [
        {
            'subject_id': 'subject001',
            'rgb_csv_path': './data/subject001/rgb_traces.csv',
            'ppg_path': './data/subject001/ppg.mat',
            'rgb_start_frame': 300
        },
        {
            'subject_id': 'subject002',
            'rgb_csv_path': './data/subject002/rgb_traces.csv',
            'ppg_path': './data/subject002/ppg.mat',
            'rgb_start_frame': 300
        }
    ]
    
    # all_data = loader.load_multiple_subjects(subject_list)
    
    print("\nData loader ready!")
    print("Modify paths in __main__ section to load your actual data")