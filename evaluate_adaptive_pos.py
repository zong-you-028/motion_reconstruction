import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import pandas as pd
import os

from adaptive_pos import AlphaPredictor, AdaptivePOS, calculate_hr_from_rppg, signal_to_noise_ratio
from data_loader import DataLoader


def evaluate_model(model, test_data, window_length=128, fs=84, device='cuda'):
    """
    Evaluate adaptive POS model on test data
    
    Args:
        model: trained AlphaPredictor
        test_data: list of dicts with 'r', 'g', 'b', 'ppg'
        window_length: POS window length
        fs: sampling frequency
        device: computation device
    
    Returns:
        results: dict with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    pos = AdaptivePOS(window_length=window_length, fs=fs)
    
    results = {
        'subject_ids': [],
        'standard_pos': {
            'correlations': [],
            'snrs': [],
            'hr_errors': [],
            'hr_predictions': [],
            'hr_ground_truth': []
        },
        'adaptive_pos': {
            'correlations': [],
            'snrs': [],
            'hr_errors': [],
            'hr_predictions': [],
            'hr_ground_truth': [],
            'alpha_mean': [],
            'alpha_std': []
        }
    }
    
    for subject_data in test_data:
        subject_id = subject_data['subject_id']
        r_buf = subject_data['r']
        g_buf = subject_data['g']
        b_buf = subject_data['b']
        ppg_gt = subject_data['ppg']
        
        print(f"Evaluating: {subject_id}")
        
        # Standard POS
        rppg_standard, alpha_std = pos.process_standard(r_buf, g_buf, b_buf)
        
        # Adaptive POS
        rppg_adaptive, alpha_adaptive = pos.process_adaptive(
            r_buf, g_buf, b_buf, model, use_features=True
        )
        
        # Remove initial transient
        valid_start = window_length
        rppg_standard = rppg_standard[valid_start:]
        rppg_adaptive = rppg_adaptive[valid_start:]
        ppg_gt_valid = ppg_gt[valid_start:]
        
        # Bandpass filter all signals
        rppg_standard_filt = bandpass_filter(rppg_standard, fs)
        rppg_adaptive_filt = bandpass_filter(rppg_adaptive, fs)
        ppg_gt_filt = bandpass_filter(ppg_gt_valid, fs)
        
        # === Standard POS Metrics ===
        # Correlation
        corr_std, _ = pearsonr(rppg_standard_filt, ppg_gt_filt)
        results['standard_pos']['correlations'].append(corr_std)
        
        # SNR
        snr_std = signal_to_noise_ratio(rppg_standard_filt, fs)
        results['standard_pos']['snrs'].append(snr_std)
        
        # Heart Rate
        try:
            hr_pred_std = calculate_hr_from_rppg(rppg_standard_filt, fs)
            hr_gt = calculate_hr_from_rppg(ppg_gt_filt, fs)
            hr_error_std = abs(hr_pred_std - hr_gt)
            
            results['standard_pos']['hr_predictions'].append(hr_pred_std)
            results['standard_pos']['hr_ground_truth'].append(hr_gt)
            results['standard_pos']['hr_errors'].append(hr_error_std)
        except:
            results['standard_pos']['hr_predictions'].append(np.nan)
            results['standard_pos']['hr_ground_truth'].append(np.nan)
            results['standard_pos']['hr_errors'].append(np.nan)
        
        # === Adaptive POS Metrics ===
        # Correlation
        corr_adapt, _ = pearsonr(rppg_adaptive_filt, ppg_gt_filt)
        results['adaptive_pos']['correlations'].append(corr_adapt)
        
        # SNR
        snr_adapt = signal_to_noise_ratio(rppg_adaptive_filt, fs)
        results['adaptive_pos']['snrs'].append(snr_adapt)
        
        # Heart Rate
        try:
            hr_pred_adapt = calculate_hr_from_rppg(rppg_adaptive_filt, fs)
            hr_error_adapt = abs(hr_pred_adapt - hr_gt)
            
            results['adaptive_pos']['hr_predictions'].append(hr_pred_adapt)
            results['adaptive_pos']['hr_ground_truth'].append(hr_gt)
            results['adaptive_pos']['hr_errors'].append(hr_error_adapt)
        except:
            results['adaptive_pos']['hr_predictions'].append(np.nan)
            results['adaptive_pos']['hr_ground_truth'].append(np.nan)
            results['adaptive_pos']['hr_errors'].append(np.nan)
        
        # Alpha statistics
        alpha_valid = alpha_adaptive[alpha_adaptive > 0]
        results['adaptive_pos']['alpha_mean'].append(np.mean(alpha_valid))
        results['adaptive_pos']['alpha_std'].append(np.std(alpha_valid))
        
        results['subject_ids'].append(subject_id)
        
        print(f"  Standard POS - Corr: {corr_std:.3f}, SNR: {snr_std:.2f} dB")
        print(f"  Adaptive POS - Corr: {corr_adapt:.3f}, SNR: {snr_adapt:.2f} dB")
        print(f"  Alpha - Mean: {np.mean(alpha_valid):.3f}, Std: {np.std(alpha_valid):.3f}")
    
    return results


def bandpass_filter(signal_data, fs, lowcut=0.7, highcut=4.0, order=4):
    """Apply bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, signal_data)
    return filtered


def plot_comparison(results, save_dir='./evaluation_results'):
    """
    Plot comparison between standard and adaptive POS
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Correlation comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results['subject_ids']))
    width = 0.35
    
    ax.bar(x - width/2, results['standard_pos']['correlations'], width, 
           label='Standard POS', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, results['adaptive_pos']['correlations'], width, 
           label='Adaptive POS', alpha=0.8, color='coral')
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('Correlation with Ground Truth')
    ax.set_title('rPPG-PPG Correlation Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(results['subject_ids'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_comparison.png'), dpi=150)
    plt.close()
    
    # 2. SNR comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, results['standard_pos']['snrs'], width, 
           label='Standard POS', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, results['adaptive_pos']['snrs'], width, 
           label='Adaptive POS', alpha=0.8, color='coral')
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Signal-to-Noise Ratio Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(results['subject_ids'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'snr_comparison.png'), dpi=150)
    plt.close()
    
    # 3. Heart Rate Error comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    hr_errors_std = [e for e in results['standard_pos']['hr_errors'] if not np.isnan(e)]
    hr_errors_adapt = [e for e in results['adaptive_pos']['hr_errors'] if not np.isnan(e)]
    
    ax.bar(x - width/2, results['standard_pos']['hr_errors'], width, 
           label='Standard POS', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, results['adaptive_pos']['hr_errors'], width, 
           label='Adaptive POS', alpha=0.8, color='coral')
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('HR Error (bpm)')
    ax.set_title('Heart Rate Estimation Error')
    ax.set_xticks(x)
    ax.set_xticklabels(results['subject_ids'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hr_error_comparison.png'), dpi=150)
    plt.close()
    
    # 4. Alpha distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(x, results['adaptive_pos']['alpha_mean'], 
                yerr=results['adaptive_pos']['alpha_std'],
                fmt='o', capsize=5, capthick=2, color='darkgreen', 
                label='Adaptive Alpha (mean ± std)')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, 
               label='Typical Standard Alpha')
    
    ax.set_xlabel('Subject')
    ax.set_ylabel('Alpha Value')
    ax.set_title('Learned Alpha Parameter Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(results['subject_ids'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'alpha_distribution.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {save_dir}/")


def print_summary_statistics(results):
    """
    Print summary statistics
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Standard POS
    print("\nSTANDARD POS:")
    print(f"  Mean Correlation: {np.mean(results['standard_pos']['correlations']):.3f} "
          f"± {np.std(results['standard_pos']['correlations']):.3f}")
    print(f"  Mean SNR: {np.mean(results['standard_pos']['snrs']):.2f} "
          f"± {np.std(results['standard_pos']['snrs']):.2f} dB")
    
    hr_errors_std = [e for e in results['standard_pos']['hr_errors'] if not np.isnan(e)]
    if hr_errors_std:
        print(f"  Mean HR Error: {np.mean(hr_errors_std):.2f} "
              f"± {np.std(hr_errors_std):.2f} bpm")
    
    # Adaptive POS
    print("\nADAPTIVE POS:")
    print(f"  Mean Correlation: {np.mean(results['adaptive_pos']['correlations']):.3f} "
          f"± {np.std(results['adaptive_pos']['correlations']):.3f}")
    print(f"  Mean SNR: {np.mean(results['adaptive_pos']['snrs']):.2f} "
          f"± {np.std(results['adaptive_pos']['snrs']):.2f} dB")
    
    hr_errors_adapt = [e for e in results['adaptive_pos']['hr_errors'] if not np.isnan(e)]
    if hr_errors_adapt:
        print(f"  Mean HR Error: {np.mean(hr_errors_adapt):.2f} "
              f"± {np.std(hr_errors_adapt):.2f} bpm")
    
    print(f"  Mean Alpha: {np.mean(results['adaptive_pos']['alpha_mean']):.3f}")
    
    # Improvements
    print("\nIMPROVEMENT (Adaptive vs Standard):")
    corr_improvement = (np.mean(results['adaptive_pos']['correlations']) - 
                       np.mean(results['standard_pos']['correlations']))
    snr_improvement = (np.mean(results['adaptive_pos']['snrs']) - 
                      np.mean(results['standard_pos']['snrs']))
    
    print(f"  Correlation: {corr_improvement:+.3f}")
    print(f"  SNR: {snr_improvement:+.2f} dB")
    
    if hr_errors_std and hr_errors_adapt:
        hr_improvement = np.mean(hr_errors_std) - np.mean(hr_errors_adapt)
        print(f"  HR Error Reduction: {hr_improvement:.2f} bpm")
    
    print("="*60)


def save_results_to_csv(results, save_path='evaluation_results.csv'):
    """
    Save results to CSV file
    """
    df_data = {
        'Subject_ID': results['subject_ids'],
        'Std_Correlation': results['standard_pos']['correlations'],
        'Std_SNR': results['standard_pos']['snrs'],
        'Std_HR_Error': results['standard_pos']['hr_errors'],
        'Adapt_Correlation': results['adaptive_pos']['correlations'],
        'Adapt_SNR': results['adaptive_pos']['snrs'],
        'Adapt_HR_Error': results['adaptive_pos']['hr_errors'],
        'Alpha_Mean': results['adaptive_pos']['alpha_mean'],
        'Alpha_Std': results['adaptive_pos']['alpha_std']
    }
    
    df = pd.DataFrame(df_data)
    df.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    print("Adaptive POS Evaluation")
    print("="*60)
    
    # Load trained model
    model_path = './adaptive_pos_models/best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_adaptive_pos.py")
        exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = AlphaPredictor(input_dim=10, hidden_dim=32)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Load test data (replace with your actual data)
    print("\nLoading test data...")
    # Example: load from saved dataset
    # test_data = load_test_data()
    
    # For demonstration, create dummy test data
    from data_loader import DataLoader
    loader = DataLoader(data_dir='./', fs=84)
    
    # Replace with your actual test subjects
    test_subjects = [
        # Add your test subject information here
    ]
    
    # test_data = loader.load_multiple_subjects(test_subjects)
    
    # Evaluate
    # results = evaluate_model(model, test_data, window_length=128, fs=84, device=device)
    
    # Visualize and save results
    # plot_comparison(results, save_dir='./evaluation_results')
    # print_summary_statistics(results)
    # save_results_to_csv(results, 'evaluation_results.csv')
    
    print("\nEvaluation script ready!")
    print("Add your test data paths and run evaluation")
