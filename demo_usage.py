"""
Data Generation Utilities for Learnable Projection POS

This script provides utility functions to generate synthetic rPPG data
for testing and training purposes.

只包含數據生成功能，不依賴其他模組。
"""

import numpy as np
from scipy import signal


def generate_synthetic_rppg_data(duration=30, fs=84, hr=75, motion_strength=0.3):
    """
    Generate synthetic rPPG data with motion artifacts
    
    Args:
        duration: duration in seconds
        fs: sampling frequency (Hz)
        hr: heart rate (bpm)
        motion_strength: strength of motion artifacts (0-1)
    
    Returns:
        r_buf, g_buf, b_buf: RGB traces (numpy arrays)
        ppg_gt: ground truth PPG signal
        hr_true: true heart rate
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # Ground truth PPG (cardiac component)
    hr_hz = hr / 60.0
    ppg_cardiac = np.sin(2 * np.pi * hr_hz * t)
    
    # Add harmonics for realistic PPG
    ppg_cardiac += 0.3 * np.sin(2 * np.pi * 2 * hr_hz * t)  # 2nd harmonic
    ppg_cardiac += 0.15 * np.sin(2 * np.pi * 3 * hr_hz * t)  # 3rd harmonic
    
    # Add respiratory modulation (0.2-0.3 Hz)
    resp_hz = 0.25
    ppg_cardiac *= (1 + 0.1 * np.sin(2 * np.pi * resp_hz * t))
    
    # Normalize
    ppg_gt = (ppg_cardiac - np.mean(ppg_cardiac)) / np.std(ppg_cardiac)
    
    # Generate RGB with PPG signal embedded
    # Base levels
    r_base = 100
    g_base = 120
    b_base = 80
    
    # PPG modulation (Green has strongest signal)
    r_modulation = 2.0 * ppg_gt
    g_modulation = 5.0 * ppg_gt  # Strongest in green
    b_modulation = 1.5 * ppg_gt
    
    # Motion artifacts (low frequency + high frequency)
    if motion_strength > 0:
        # Low frequency drift (0.05-0.5 Hz)
        motion_lf = motion_strength * 10 * (
            np.sin(2 * np.pi * 0.1 * t) + 
            0.5 * np.sin(2 * np.pi * 0.3 * t)
        )
        
        # High frequency noise (> 4 Hz)
        motion_hf = motion_strength * 5 * np.random.randn(n_samples)
        
        # Sudden spikes (simulating rapid head movements)
        n_spikes = int(motion_strength * 10)
        spike_indices = np.random.choice(n_samples, n_spikes, replace=False)
        spikes = np.zeros(n_samples)
        for idx in spike_indices:
            spike_width = int(0.2 * fs)  # 0.2 second spikes
            start = max(0, idx - spike_width // 2)
            end = min(n_samples, idx + spike_width // 2)
            spikes[start:end] = motion_strength * 20 * np.exp(-((np.arange(end - start) - spike_width // 2) ** 2) / (spike_width / 4) ** 2)
        
        motion = motion_lf + motion_hf + spikes
    else:
        motion = np.zeros(n_samples)
    
    # Add Gaussian noise
    noise_level = 2.0
    r_noise = noise_level * np.random.randn(n_samples)
    g_noise = noise_level * np.random.randn(n_samples)
    b_noise = noise_level * np.random.randn(n_samples)
    
    # Combine all components
    r_buf = r_base + r_modulation + motion + r_noise
    g_buf = g_base + g_modulation + motion + g_noise
    b_buf = b_base + b_modulation + motion + b_noise
    
    # Ensure positive values
    r_buf = np.maximum(r_buf, 1.0)
    g_buf = np.maximum(g_buf, 1.0)
    b_buf = np.maximum(b_buf, 1.0)
    
    return r_buf, g_buf, b_buf, ppg_gt, hr


def generate_batch_synthetic_data(n_subjects=20, duration=30, fs=84, 
                                   hr_range=(60, 100), motion_range=(0.2, 0.6)):
    """
    Generate a batch of synthetic rPPG data for multiple subjects
    
    Args:
        n_subjects: number of subjects to generate
        duration: duration in seconds
        fs: sampling frequency
        hr_range: tuple of (min_hr, max_hr)
        motion_range: tuple of (min_motion, max_motion)
    
    Returns:
        rgb_traces: list of (r, g, b) tuples
        ppg_signals: list of PPG arrays
        heart_rates: list of true heart rates
    """
    rgb_traces = []
    ppg_signals = []
    heart_rates = []
    
    for i in range(n_subjects):
        # Random parameters
        hr = np.random.uniform(hr_range[0], hr_range[1])
        motion = np.random.uniform(motion_range[0], motion_range[1])
        
        # Generate data
        r, g, b, ppg, _ = generate_synthetic_rppg_data(
            duration=duration,
            fs=fs,
            hr=hr,
            motion_strength=motion
        )
        
        rgb_traces.append((r, g, b))
        ppg_signals.append(ppg)
        heart_rates.append(hr)
    
    return rgb_traces, ppg_signals, heart_rates


def add_realistic_artifacts(r_buf, g_buf, b_buf, artifact_type='blink', strength=1.0):
    """
    Add specific realistic artifacts to RGB traces
    
    Args:
        r_buf, g_buf, b_buf: RGB traces
        artifact_type: 'blink', 'head_movement', 'lighting_change'
        strength: artifact strength (0-1)
    
    Returns:
        modified r_buf, g_buf, b_buf
    """
    r_buf = r_buf.copy()
    g_buf = g_buf.copy()
    b_buf = b_buf.copy()
    
    n_samples = len(r_buf)
    
    if artifact_type == 'blink':
        # Simulate eye blinks (brief RGB drops)
        n_blinks = int(strength * 5)
        for _ in range(n_blinks):
            blink_pos = np.random.randint(0, n_samples - 10)
            blink_duration = np.random.randint(3, 8)
            drop = strength * 0.5
            r_buf[blink_pos:blink_pos + blink_duration] *= (1 - drop)
            g_buf[blink_pos:blink_pos + blink_duration] *= (1 - drop)
            b_buf[blink_pos:blink_pos + blink_duration] *= (1 - drop)
    
    elif artifact_type == 'head_movement':
        # Simulate head rotation (RGB channel imbalance)
        movement_pos = np.random.randint(0, n_samples - 50)
        movement_duration = np.random.randint(30, 100)
        
        # Gradual change
        transition = np.linspace(0, 1, movement_duration)
        r_change = strength * 10 * np.sin(np.pi * transition)
        g_change = -strength * 8 * np.sin(np.pi * transition)
        b_change = strength * 5 * np.sin(np.pi * transition)
        
        r_buf[movement_pos:movement_pos + movement_duration] += r_change
        g_buf[movement_pos:movement_pos + movement_duration] += g_change
        b_buf[movement_pos:movement_pos + movement_duration] += b_change
    
    elif artifact_type == 'lighting_change':
        # Simulate lighting variation
        change_pos = np.random.randint(0, n_samples - 100)
        
        # Sudden step change
        scale = 1 + strength * 0.3
        r_buf[change_pos:] *= scale
        g_buf[change_pos:] *= scale
        b_buf[change_pos:] *= scale
    
    return r_buf, g_buf, b_buf


def calculate_snr(signal_data, fs, hr_range=(0.7, 4.0)):
    """
    Calculate Signal-to-Noise Ratio in the heart rate frequency band
    
    Args:
        signal_data: 1D signal array
        fs: sampling frequency
        hr_range: heart rate frequency range in Hz
    
    Returns:
        snr_db: SNR in dB
    """
    # Compute FFT
    n = len(signal_data)
    fft_vals = np.fft.fft(signal_data)
    fft_freq = np.fft.fftfreq(n, 1/fs)
    
    # Only positive frequencies
    pos_mask = fft_freq > 0
    fft_vals = np.abs(fft_vals[pos_mask])
    fft_freq = fft_freq[pos_mask]
    
    # Signal power in HR band
    hr_mask = (fft_freq >= hr_range[0]) & (fft_freq <= hr_range[1])
    signal_power = np.sum(fft_vals[hr_mask] ** 2)
    
    # Noise power outside HR band
    noise_mask = ~hr_mask
    noise_power = np.sum(fft_vals[noise_mask] ** 2)
    
    # SNR in dB
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    
    return snr_db


if __name__ == "__main__":
    print("數據生成工具測試")
    print("=" * 60)
    
    # 測試 1: 單個受試者
    print("\n1. 生成單個受試者數據...")
    r, g, b, ppg, hr = generate_synthetic_rppg_data(
        duration=30, fs=84, hr=75, motion_strength=0.3
    )
    print(f"   ✓ 生成 {len(r)} 個樣本")
    print(f"   真實心率: {hr} bpm")
    print(f"   RGB 範圍: R={r.min():.1f}-{r.max():.1f}, "
          f"G={g.min():.1f}-{g.max():.1f}, B={b.min():.1f}-{b.max():.1f}")
    
    # 測試 2: 批次生成
    print("\n2. 批次生成數據...")
    rgb_traces, ppg_signals, hrs = generate_batch_synthetic_data(
        n_subjects=10, duration=20, fs=84
    )
    print(f"   ✓ 生成 {len(rgb_traces)} 個受試者")
    print(f"   心率範圍: {min(hrs):.1f} - {max(hrs):.1f} bpm")
    
    # 測試 3: 添加偽影
    print("\n3. 添加真實偽影...")
    r2, g2, b2 = add_realistic_artifacts(r, g, b, 'blink', strength=0.8)
    print(f"   ✓ 添加眨眼偽影")
    
    r3, g3, b3 = add_realistic_artifacts(r, g, b, 'head_movement', strength=0.5)
    print(f"   ✓ 添加頭部運動偽影")
    
    # 測試 4: 計算 SNR
    print("\n4. 計算 SNR...")
    snr = calculate_snr(ppg, fs=84)
    print(f"   ✓ SNR: {snr:.2f} dB")
    
    print("\n✓ 所有測試完成！")
    print("\n這個模組可以被其他程式導入使用：")
    print("  from demo_usage import generate_synthetic_rppg_data")