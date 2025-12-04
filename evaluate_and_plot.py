"""
evaluate_and_plot.py
評估模型與繪製圖表的專用模組
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy import signal

from learnable_projection_pos import LearnableProjectionPOS

# 設定中文字體（避免方塊），優先使用微軟正黑體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_hr_from_rppg(rppg_signal, fs):
    """
    從 rPPG 訊號計算心率
    """
    # FFT
    n = len(rppg_signal)
    fft_vals = np.fft.fft(rppg_signal)
    fft_freq = np.fft.fftfreq(n, 1/fs)
    
    # 只取正頻率
    pos_mask = (fft_freq > 0) & (fft_freq < 5)  # 0-5 Hz (0-300 bpm)
    fft_freq = fft_freq[pos_mask]
    fft_vals = np.abs(fft_vals[pos_mask])
    
    # 限制在合理的心率範圍 (0.7-4.0 Hz = 42-240 bpm)
    hr_mask = (fft_freq >= 0.7) & (fft_freq <= 4.0)
    
    if not np.any(hr_mask):
        return 0
    
    # 找到峰值
    hr_freq = fft_freq[hr_mask]
    hr_fft = fft_vals[hr_mask]
    
    peak_idx = np.argmax(hr_fft)
    hr_hz = hr_freq[peak_idx]
    hr_bpm = hr_hz * 60
    
    return hr_bpm


def evaluate_single_subject(model, rgb_traces, ppg_gt, subject_id, fs=30, device='cpu'):
    """
    評估單個受試者
    """
    r, g, b = rgb_traces
    
    # 使用模型處理
    pos = LearnableProjectionPOS(window_length=128, fs=fs)
    
    # 標準 POS
    rppg_standard = pos.process_standard(r, g, b)
    
    # 可學習投影矩陣 POS
    model.eval()
    with torch.no_grad():
        rppg_learned, P_history = pos.process_learnable(r, g, b, model, use_features=True)
    
    # 去除初始段（避免邊界效應）
    valid_start = 128
    rppg_standard = rppg_standard[valid_start:]
    rppg_learned = rppg_learned[valid_start:]
    ppg_gt = ppg_gt[valid_start:]
    
    # 確保長度一致
    min_len = min(len(rppg_standard), len(rppg_learned), len(ppg_gt))
    rppg_standard = rppg_standard[:min_len]
    rppg_learned = rppg_learned[:min_len]
    ppg_gt = ppg_gt[:min_len]
    
    # 帶通濾波（心率範圍 0.7-4.0 Hz）
    nyq = fs / 2
    b_filt, a_filt = signal.butter(4, [0.7/nyq, 4.0/nyq], btype='band')
    
    rppg_std_filt = signal.filtfilt(b_filt, a_filt, rppg_standard)
    rppg_learn_filt = signal.filtfilt(b_filt, a_filt, rppg_learned)
    ppg_gt_filt = signal.filtfilt(b_filt, a_filt, ppg_gt)
    
    # 計算指標
    # Correlation
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
    
    # Heart Rate
    hr_gt = calculate_hr_from_rppg(ppg_gt_filt, fs)
    hr_std = calculate_hr_from_rppg(rppg_std_filt, fs)
    hr_learn = calculate_hr_from_rppg(rppg_learn_filt, fs)
    
    hr_error_std = abs(hr_std - hr_gt)
    hr_error_learn = abs(hr_learn - hr_gt)
    
    return {
        'subject_id': subject_id,
        'signals': {
            'rppg_standard': rppg_std_filt,
            'rppg_learned': rppg_learn_filt,
            'ppg_gt': ppg_gt_filt,
            'time': np.arange(len(ppg_gt_filt)) / fs
        },
        'metrics': {
            'corr_std': corr_std,
            'corr_learn': corr_learn,
            'snr_std': snr_std,
            'snr_learn': snr_learn,
            'hr_gt': hr_gt,
            'hr_std': hr_std,
            'hr_learn': hr_learn,
            'hr_error_std': hr_error_std,
            'hr_error_learn': hr_error_learn
        },
        'P_history': P_history
    }


def plot_single_subject_results(results, save_path):
    """
    繪製單個受試者的結果圖
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    subject_id = results['subject_id']
    signals = results['signals']
    metrics = results['metrics']
    P_history = results['P_history']
    
    # 1. 訊號波形對比（上半部）
    ax1 = fig.add_subplot(gs[0, :])
    t = signals['time']
    
    # 只顯示前 10 秒（更清楚）
    display_duration = 10  # 秒
    display_samples = int(display_duration * 30)
    if display_samples > len(t):
        display_samples = len(t)
    
    ax1.plot(t[:display_samples], signals['ppg_gt'][:display_samples], 
             'k-', linewidth=2, label='Ground Truth PPG', alpha=0.7)
    ax1.plot(t[:display_samples], signals['rppg_standard'][:display_samples], 
             'b--', linewidth=1.5, label='Standard POS', alpha=0.7)
    ax1.plot(t[:display_samples], signals['rppg_learned'][:display_samples], 
             'r-', linewidth=1.5, label='Learned Projection POS', alpha=0.8)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Amplitude (normalized)', fontsize=12)
    ax1.set_title(f'{subject_id} - rPPG Signal Comparison (First 10s)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 頻譜對比
    ax2 = fig.add_subplot(gs[1, 0])
    
    def plot_spectrum(sig, fs, label, color, linestyle='-'):
        n = len(sig)
        fft_vals = np.fft.fft(sig)
        fft_freq = np.fft.fftfreq(n, 1/fs)
        
        pos_mask = (fft_freq > 0) & (fft_freq < 5)
        fft_freq = fft_freq[pos_mask]
        fft_vals = np.abs(fft_vals[pos_mask])
        
        ax2.plot(fft_freq * 60, fft_vals, linestyle, linewidth=2, 
                 label=label, color=color, alpha=0.7)
    
    plot_spectrum(signals['ppg_gt'], 30, 'Ground Truth', 'black', '-')
    plot_spectrum(signals['rppg_standard'], 30, 'Standard POS', 'blue', '--')
    plot_spectrum(signals['rppg_learned'], 30, 'Learned Projection', 'red', '-')
    
    ax2.set_xlabel('Heart Rate (bpm)', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([40, 180])
    
    # 3. 指標對比（柱狀圖）
    ax3 = fig.add_subplot(gs[1, 1])
    
    metrics_names = ['Correlation', 'SNR (dB)', 'HR Error (bpm)']
    std_vals = [metrics['corr_std'], metrics['snr_std'], metrics['hr_error_std']]
    learn_vals = [metrics['corr_learn'], metrics['snr_learn'], metrics['hr_error_learn']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, std_vals, width, label='Standard POS', color='skyblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, learn_vals, width, label='Learned Projection', color='salmon', alpha=0.8)
    
    ax3.set_xlabel('Metrics', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_names, fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 在柱上顯示數值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 4. 心率對比
    ax4 = fig.add_subplot(gs[2, 0])
    
    hr_data = [metrics['hr_gt'], metrics['hr_std'], metrics['hr_learn']]
    hr_labels = ['Ground Truth', 'Standard POS', 'Learned Projection']
    colors = ['black', 'skyblue', 'salmon']
    
    bars = ax4.barh(hr_labels, hr_data, color=colors, alpha=0.7)
    ax4.set_xlabel('Heart Rate (bpm)', fontsize=12)
    ax4.set_title('Heart Rate Estimation', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, hr_data)):
        ax4.text(val, i, f' {val:.1f} bpm', va='center', fontsize=10, fontweight='bold')
    
    # 5. 投影矩陣演化（6 個子圖）
    P_standard = np.array([[0, 1, -1], [-2, 1, 1]])
    
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[2 + i, 1])
            if i == 0 and j == 0:  # 只在第一個子圖顯示
                ax.plot(P_history[:, i, j], 'g-', linewidth=2, label='Learned', alpha=0.8)
                ax.axhline(P_standard[i, j], color='r', linestyle='--', 
                           linewidth=2, label='Standard', alpha=0.8)
                ax.legend(fontsize=8, loc='upper right')
            else:
                ax.plot(P_history[:, i, j], 'g-', linewidth=2, alpha=0.8)
                ax.axhline(P_standard[i, j], color='r', linestyle='--', linewidth=2, alpha=0.8)
            
            ax.set_title(f'P[{i},{j}]', fontsize=10, fontweight='bold')
            ax.set_xlabel('Time Window', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
    
    # 隱藏多餘的子圖
    for idx in range(2, 6):
        ax = fig.add_subplot(gs[2 + idx//3, 1])
        if idx >= 2:
            fig.delaxes(ax)
    
    # 6. 統計摘要（文字框）
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    summary_text = f"""
    Subject: {subject_id}
    ──────────────────────────────────────────────────────────────────────────
                            Standard POS          Learned Projection          Improvement
    ──────────────────────────────────────────────────────────────────────────
    Correlation              {metrics['corr_std']:8.4f}              {metrics['corr_learn']:8.4f}              {(metrics['corr_learn'] - metrics['corr_std'])*100:+7.2f}%
    SNR (dB)                 {metrics['snr_std']:8.2f}              {metrics['snr_learn']:8.2f}              {(metrics['snr_learn'] - metrics['snr_std']):+7.2f} dB
    Heart Rate (bpm)         {metrics['hr_std']:8.1f}              {metrics['hr_learn']:8.1f}              GT: {metrics['hr_gt']:.1f}
    HR Error (bpm)           {metrics['hr_error_std']:8.2f}              {metrics['hr_error_learn']:8.2f}              {(metrics['hr_error_std'] - metrics['hr_error_learn']):+7.2f}
    ──────────────────────────────────────────────────────────────────────────
    """
    
    ax5.text(0.5, 0.5, summary_text, ha='center', va='center', 
             fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Learnable Projection POS - Evaluation Results\n{subject_id}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已保存: {save_path}")
    plt.close()


def plot_overall_summary(all_results, save_path):
    """
    繪製所有受試者的總結圖
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Learnable Projection POS - Overall Performance Summary', 
                 fontsize=16, fontweight='bold')
    
    # 提取所有指標
    subjects = [r['subject_id'] for r in all_results]
    corr_std = [r['metrics']['corr_std'] for r in all_results]
    corr_learn = [r['metrics']['corr_learn'] for r in all_results]
    snr_std = [r['metrics']['snr_std'] for r in all_results]
    snr_learn = [r['metrics']['snr_learn'] for r in all_results]
    hr_error_std = [r['metrics']['hr_error_std'] for r in all_results]
    hr_error_learn = [r['metrics']['hr_error_learn'] for r in all_results]
    
    # 1. Correlation 對比
    ax = axes[0, 0]
    x = np.arange(len(subjects))
    width = 0.35
    ax.bar(x - width/2, corr_std, width, label='Standard POS', color='skyblue', alpha=0.7)
    ax.bar(x + width/2, corr_learn, width, label='Learned Projection', color='salmon', alpha=0.7)
    ax.set_xlabel('Subject ID', fontsize=11)
    ax.set_ylabel('Correlation', fontsize=11)
    ax.set_title('Correlation Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 2. SNR 對比
    ax = axes[0, 1]
    ax.bar(x - width/2, snr_std, width, label='Standard POS', color='skyblue', alpha=0.7)
    ax.bar(x + width/2, snr_learn, width, label='Learned Projection', color='salmon', alpha=0.7)
    ax.set_xlabel('Subject ID', fontsize=11)
    ax.set_ylabel('SNR (dB)', fontsize=11)
    ax.set_title('SNR Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 3. HR Error 對比
    ax = axes[0, 2]
    ax.bar(x - width/2, hr_error_std, width, label='Standard POS', color='skyblue', alpha=0.7)
    ax.bar(x + width/2, hr_error_learn, width, label='Learned Projection', color='salmon', alpha=0.7)
    ax.set_xlabel('Subject ID', fontsize=11)
    ax.set_ylabel('HR Error (bpm)', fontsize=11)
    ax.set_title('Heart Rate Error Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 4. Correlation 改善百分比
    ax = axes[1, 0]
    corr_improvement = [(l - s) / s * 100 for s, l in zip(corr_std, corr_learn)]
    colors = ['green' if x > 0 else 'red' for x in corr_improvement]
    ax.bar(x, corr_improvement, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Subject ID', fontsize=11)
    ax.set_ylabel('Improvement (%)', fontsize=11)
    ax.set_title('Correlation Improvement', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 5. SNR 改善
    ax = axes[1, 1]
    snr_improvement = [l - s for s, l in zip(snr_std, snr_learn)]
    colors = ['green' if x > 0 else 'red' for x in snr_improvement]
    ax.bar(x, snr_improvement, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Subject ID', fontsize=11)
    ax.set_ylabel('Improvement (dB)', fontsize=11)
    ax.set_title('SNR Improvement', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 6. 統計摘要
    ax = axes[1, 2]
    ax.axis('off')
    
    mean_corr_std = np.mean(corr_std)
    mean_corr_learn = np.mean(corr_learn)
    mean_snr_std = np.mean(snr_std)
    mean_snr_learn = np.mean(snr_learn)
    mean_hr_error_std = np.mean(hr_error_std)
    mean_hr_error_learn = np.mean(hr_error_learn)
    
    summary_text = f"""
    Overall Statistics ({len(subjects)} subjects)
    
    ──────────────────────────────────────
    Metric              Standard    Learned
    ──────────────────────────────────────
    Correlation (mean)  {mean_corr_std:.4f}    {mean_corr_learn:.4f}
    SNR (mean, dB)      {mean_snr_std:.2f}      {mean_snr_learn:.2f}
    HR Error (mean, bpm) {mean_hr_error_std:.2f}       {mean_hr_error_learn:.2f}
    ──────────────────────────────────────
    
    Improvement:
    • Correlation: {(mean_corr_learn - mean_corr_std) / mean_corr_std * 100:+.2f}%
    • SNR: {(mean_snr_learn - mean_snr_std):+.2f} dB
    • HR Error: {(mean_hr_error_std - mean_hr_error_learn):+.2f} bpm
    """
    
    ax.text(0.1, 0.5, summary_text, ha='left', va='center',
            fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ 總結圖已保存: {save_path}")
    plt.close()