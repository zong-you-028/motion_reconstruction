"""
evaluate_and_plot.py
評估模型與繪製圖表的專用模組 (修復版面配置)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from scipy import signal

from learnable_projection_pos import LearnableProjectionPOS

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def normalize_to_neg1_1(sig):
    """將訊號標準化到 [-1, 1] 範圍"""
    sig_min = np.min(sig)
    sig_max = np.max(sig)
    if sig_max - sig_min > 1e-6:
        return 2 * (sig - sig_min) / (sig_max - sig_min) - 1
    return sig


def calculate_hr_from_rppg(rppg_signal, fs):
    """從 rPPG 訊號計算心率"""
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


def evaluate_single_subject(model, rgb_traces, ppg_gt, subject_id, fs=30, device='cpu'):
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
    
    # 對齊長度
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
    
    # Normalize 到 [-1, 1] 以便比較
    rppg_std_filt = normalize_to_neg1_1(rppg_std_filt)
    rppg_learn_filt = normalize_to_neg1_1(rppg_learn_filt)
    ppg_gt_filt = normalize_to_neg1_1(ppg_gt_filt)
    
    # 計算指標
    corr_std, _ = pearsonr(rppg_std_filt, ppg_gt_filt)
    corr_learn, _ = pearsonr(rppg_learn_filt, ppg_gt_filt)
    
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
            'corr_std': corr_std, 'corr_learn': corr_learn,
            'snr_std': snr_std, 'snr_learn': snr_learn,
            'hr_gt': hr_gt, 'hr_std': hr_std, 'hr_learn': hr_learn,
            'hr_error_std': hr_error_std, 'hr_error_learn': hr_error_learn
        },
        'P_history': P_history
    }


def plot_single_subject_results(results, save_path):
    """繪製單個受試者的結果圖 (版面優化版)"""
    # 增加圖表高度，給下方 Summary 更多空間
    fig = plt.figure(figsize=(16, 14))
    
    # 調整 gridspec 高度比例，加大最後一列(Summary)的高度
    gs = fig.add_gridspec(5, 2, height_ratios=[1.5, 1, 1, 1, 0.8], hspace=0.4, wspace=0.25)
    
    subject_id = results['subject_id']
    signals = results['signals']
    metrics = results['metrics']
    P_history = results['P_history']
    
    # 1. 訊號波形對比
    ax1 = fig.add_subplot(gs[0, :])
    t = signals['time']
    display_duration = 10 
    display_samples = int(display_duration * 30)
    if display_samples > len(t): display_samples = len(t)
    
    ax1.plot(t[:display_samples], signals['ppg_gt'][:display_samples], 'k-', linewidth=2, label='Ground Truth', alpha=0.6)
    ax1.plot(t[:display_samples], signals['rppg_standard'][:display_samples], 'b--', linewidth=1.5, label='Standard POS', alpha=0.6)
    ax1.plot(t[:display_samples], signals['rppg_learned'][:display_samples], 'r-', linewidth=1.5, label='Learned POS', alpha=0.8)
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (norm)')
    ax1.set_title(f'Signal Comparison (First 10s) - {subject_id}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 頻譜對比
    ax2 = fig.add_subplot(gs[1, 0])
    def plot_spectrum(sig, fs, label, color, style='-'):
        n = len(sig)
        fft_vals = np.fft.fft(sig)
        fft_freq = np.fft.fftfreq(n, 1/fs)
        pos_mask = (fft_freq > 0) & (fft_freq < 5)
        ax2.plot(fft_freq[pos_mask]*60, np.abs(fft_vals[pos_mask]), style, color=color, label=label, alpha=0.7)
    
    plot_spectrum(signals['ppg_gt'], 30, 'GT', 'black')
    plot_spectrum(signals['rppg_standard'], 30, 'Std POS', 'blue', '--')
    plot_spectrum(signals['rppg_learned'], 30, 'Learned', 'red')
    ax2.set_xlabel('HR (bpm)')
    ax2.set_ylabel('Mag')
    ax2.set_title('Frequency Spectrum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([40, 180])
    
    # 3. 指標對比
    ax3 = fig.add_subplot(gs[1, 1])
    names = ['Corr', 'SNR (dB)', 'HR Err']
    std_v = [metrics['corr_std'], metrics['snr_std'], metrics['hr_error_std']]
    lrn_v = [metrics['corr_learn'], metrics['snr_learn'], metrics['hr_error_learn']]
    x = np.arange(3); w = 0.35
    ax3.bar(x-w/2, std_v, w, label='Std POS', color='skyblue')
    ax3.bar(x+w/2, lrn_v, w, label='Learned', color='salmon')
    ax3.set_xticks(x); ax3.set_xticklabels(names)
    ax3.set_title('Metrics')
    ax3.legend()
    ax3.grid(True, axis='y', alpha=0.3)
    
    # 4. 心率估計
    ax4 = fig.add_subplot(gs[2, 0])
    vals = [metrics['hr_gt'], metrics['hr_std'], metrics['hr_learn']]
    labels = ['GT', 'Std POS', 'Learned']
    ax4.barh(labels, vals, color=['black', 'skyblue', 'salmon'], alpha=0.7)
    ax4.set_xlabel('BPM')
    ax4.set_title('Heart Rate')
    for i, v in enumerate(vals):
        ax4.text(v, i, f' {v:.1f}', va='center')
        
    # 5. 投影矩陣變化 (簡化顯示)
    # 只畫 P[0,1] (Green weight for S1) 和 P[1,1] (Green weight for S2)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(P_history[:, 0, 1], label='P[0,1] (G)', color='green', alpha=0.7)
    ax5.plot(P_history[:, 0, 2], label='P[0,2] (B)', color='blue', alpha=0.7)
    ax5.set_title('Projection Vector 1 (Weights)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.plot(P_history[:, 1, 1], label='P[1,1] (G)', color='green', alpha=0.7)
    ax6.plot(P_history[:, 1, 2], label='P[1,2] (B)', color='blue', alpha=0.7)
    ax6.set_title('Projection Vector 2 (Weights)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 6. Summary 文字框 (移到最下方，避免重疊)
    ax_sum = fig.add_subplot(gs[4, :])
    ax_sum.axis('off')
    txt = f"Subject: {subject_id} | Corr: {metrics['corr_learn']:.3f} | SNR: {metrics['snr_learn']:.1f}dB | HR Err: {metrics['hr_error_learn']:.1f}bpm"
    ax_sum.text(0.5, 0.5, txt, ha='center', va='center', fontsize=12, bbox=dict(facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Evaluation Results: {subject_id}', fontsize=16, fontweight='bold', y=0.99)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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