import os
import numpy as np
import pandas as pd
from scipy import signal, io
import matplotlib.pyplot as plt

class DataLoader:
    """
    Load RGB traces from FaceMesh extraction and PPG ground truth
    """
    def __init__(self, data_dir, fs=30):
        self.data_dir = data_dir
        self.fs = fs
        
    def load_rgb_from_csv(self, csv_path):
        """Load RGB traces from FaceMesh output CSV"""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV {csv_path}: {e}")
        
        required_cols = ['R_avg', 'G_avg', 'B_avg']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 簡單過濾掉明顯錯誤的幀 (全0)
        if 'success' in df.columns:
            valid_mask = (df['R_avg'] > 0) | (df['G_avg'] > 0) | (df['B_avg'] > 0)
            df = df[valid_mask]
        
        if len(df) == 0:
            raise ValueError(f"No valid frames in {csv_path}")
        
        r_trace = df['R_avg'].values
        g_trace = df['G_avg'].values
        b_trace = df['B_avg'].values
        
        return r_trace, g_trace, b_trace
    
    def load_ppg_from_mat(self, mat_path, signal_name='ppg'):
        """Load PPG from .mat file"""
        try:
            mat_data = io.loadmat(mat_path)
        except Exception as e:
            raise ValueError(f"Error reading MAT {mat_path}: {e}")
        
        if signal_name in mat_data:
            ppg = mat_data[signal_name].flatten()
        else:
            # 自動尋找可能的變數名稱
            possible_names = ['ppg', 'PPG', 'signal', 'data', 'BVP', 'pulse']
            for name in possible_names:
                if name in mat_data:
                    ppg = mat_data[name].flatten()
                    break
                for key in mat_data.keys():
                    if key.lower() == name.lower():
                        ppg = mat_data[key].flatten()
                        break
                else:
                    continue
                break
            else:
                # 使用最大的陣列
                best_key = None
                max_len = 0
                for key, val in mat_data.items():
                    if isinstance(val, np.ndarray) and val.size > max_len and not key.startswith('__'):
                        max_len = val.size
                        best_key = key
                
                if best_key:
                    ppg = mat_data[best_key].flatten()
                else:
                    raise ValueError(f"Could not find PPG signal in {mat_path}")
        
        return ppg
    
    def load_ppg_from_csv(self, csv_path, ppg_column='PPG'):
        """Load PPG from CSV file"""
        try:
            df_peek = pd.read_csv(csv_path, nrows=5)
            has_header = True
            try:
                float(df_peek.columns[0])
                has_header = False
            except:
                pass
            
            if has_header:
                df = pd.read_csv(csv_path)
                target_col = None
                candidates = [ppg_column, 'ppg', 'PPG', 'BVP', 'bvp', 'signal', 'Signal']
                
                for col in df.columns:
                    if col in candidates:
                        target_col = col
                        break
                if target_col is None:
                    for col in df.columns:
                        if isinstance(col, str) and 'ppg' in col.lower():
                            target_col = col
                            break
                if target_col is None:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        target_col = numeric_cols[-1]
                
                if target_col:
                    ppg = df[target_col].values
                else:
                    ppg = df.iloc[:, 1].values if df.shape[1] > 1 else df.iloc[:, 0].values
            else:
                df = pd.read_csv(csv_path, header=None)
                ppg = df.iloc[:, -1].values
                
        except Exception as e:
            raise ValueError(f"Error reading PPG CSV {csv_path}: {e}")
            
        return ppg
    
    def align_rgb_ppg(self, r_trace, g_trace, b_trace, ppg, 
                      rgb_start_frame=0, rgb_fps=None, ppg_fps=None):
        """Align RGB and PPG"""
        if rgb_fps is None: rgb_fps = self.fs
        if ppg_fps is None: ppg_fps = self.fs
        
        min_len = min(len(r_trace), len(ppg))
        
        r_trace = r_trace[:min_len]
        g_trace = g_trace[:min_len]
        b_trace = b_trace[:min_len]
        ppg = ppg[:min_len]
        
        return r_trace, g_trace, b_trace, ppg
    
    def preprocess_signals(self, r_trace, g_trace, b_trace, ppg):
        """
        Preprocess RGB and PPG signals
        
        策略：
        1. RGB: 保持原始數值 (Raw Intensity)，不要去均值或標準化。
           原因: POS 演算法公式依賴 I(t)/mean(I)，若 mean(I)=0 會導致除以零。
        
        2. PPG: Normalize 到 [-1, 1]。
           原因: 讓 Ground Truth 在固定範圍內，有助於模型收斂。
        """
        # 1. 處理無效值
        r_trace = np.nan_to_num(r_trace)
        g_trace = np.nan_to_num(g_trace)
        b_trace = np.nan_to_num(b_trace)
        ppg = np.nan_to_num(ppg)
        
        # 2. RGB 處理: 保持原始數值 (不做任何處理)
        
        # 3. PPG 處理: 去趨勢 + Min-Max Normalization 到 [-1, 1]
        ppg = signal.detrend(ppg)
        
        ppg_min = np.min(ppg)
        ppg_max = np.max(ppg)
        
        if ppg_max - ppg_min > 1e-6:
            # 公式: 2 * (x - min) / (max - min) - 1
            ppg = 2 * (ppg - ppg_min) / (ppg_max - ppg_min) - 1
        else:
            ppg[:] = 0
            
        return r_trace, g_trace, b_trace, ppg
    
    def load_subject_data(self, subject_id, rgb_csv_path, ppg_path, 
                         rgb_start_frame=0, visualize=False):
        """Load complete data for one subject"""
        # Load
        r_trace, g_trace, b_trace = self.load_rgb_from_csv(rgb_csv_path)
        
        if ppg_path.endswith('.mat'):
            ppg = self.load_ppg_from_mat(ppg_path)
        else:
            ppg = self.load_ppg_from_csv(ppg_path)
            
        # Align
        r_trace, g_trace, b_trace, ppg = self.align_rgb_ppg(
            r_trace, g_trace, b_trace, ppg, 
            rgb_start_frame=rgb_start_frame,
            rgb_fps=self.fs,
            ppg_fps=self.fs
        )
        
        # Preprocess (Raw RGB, Normalized PPG)
        r_trace, g_trace, b_trace, ppg = self.preprocess_signals(
            r_trace, g_trace, b_trace, ppg
        )
        
        return {
            'subject_id': subject_id,
            'r': r_trace,
            'g': g_trace,
            'b': b_trace,
            'ppg': ppg,
            'fs': self.fs
        }