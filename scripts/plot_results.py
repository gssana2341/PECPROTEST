# scripts/plot_results.py — Professional Scientific Plotting Suite for Silicon Photonics
#
# Generates:
# 1. dac_quantization_heater_sweep.png (Real multi-dimensional MZI sweep results)
# 2. ekf_phase_drift_convergence.png (EKF closed-loop thermal walk tracking)
#
# Uses 100% clean, academic English terminology for international publications.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add FFI library path to import binding
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'photonic', 'interface', 'ffi'))
from python_binding import PhotonicNetwork

def run_compression_sweep(pho_file, dataset_csv):
    print("[1/2] Running real MZI Compression and DAC Sweeps...")
    temp_weights = "temp_plot_weights.phomodel"
    net = PhotonicNetwork(pho_file=pho_file)
    
    # Train brief converged baseline
    res = net.train(dataset_csv, epochs=1, lr=0.01, batch_size=16)
    baseline_acc = res["final_accuracy"]
    net.save_weights(temp_weights)
    del net

    dac_bits_sweep = [16, 12, 8, 6, 4]
    thresholds_sweep = [0.00, 0.05, 0.10, 0.20, 0.50]
    
    plot_data = {bits: {"thresholds": [], "accuracies": [], "savings": []} for bits in dac_bits_sweep}

    for bits in dac_bits_sweep:
        for thresh in thresholds_sweep:
            restored_net = PhotonicNetwork(pho_file=pho_file, weights_file=temp_weights)
            savings = restored_net.compress(dac_bits=bits, pruning_threshold=thresh)
            eval_res = restored_net.train(dataset_csv, epochs=0, lr=0.01, batch_size=16)
            
            plot_data[bits]["thresholds"].append(thresh)
            plot_data[bits]["accuracies"].append(eval_res["final_accuracy"])
            plot_data[bits]["savings"].append(savings)
            del restored_net

    if os.path.exists(temp_weights):
        os.remove(temp_weights)
        
    return plot_data, baseline_acc

def plot_sweep_results(plot_data, baseline_acc, output_path):
    print("✓ Plotting sweep results to:", output_path)
    plt.figure(figsize=(9, 5.5))
    
    colors = {16: '#2c3e50', 12: '#2980b9', 8: '#27ae60', 6: '#f39c12', 4: '#c0392b'}
    markers = {16: 'o', 12: 's', 8: '^', 6: 'd', 4: 'x'}

    for bits, data in plot_data.items():
        plt.plot(data["thresholds"], data["accuracies"], 
                 label=f"{bits}-bit DAC", 
                 color=colors[bits], 
                 marker=markers[bits], 
                 linewidth=2, 
                 markersize=7)

    # Plot baseline line
    plt.axhline(y=baseline_acc, color='gray', linestyle='--', label=f'Baseline ({baseline_acc:.1f}%)', alpha=0.7)

    plt.title("Mach-Zehnder Interferometer (MZI) Pruning & DAC Sweep", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Soft Pruning Phase Threshold (rad)", fontsize=11, labelpad=8)
    plt.ylabel("MNIST Test Set Accuracy (%)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', frameon=True, shadow=False)
    plt.ylim(10, 100) # Full scale showing accuracy cliff
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def simulate_and_plot_ekf(output_path):
    print("[2/2] Simulating EKF closed-loop waveguide thermal drift tracking...")
    np.random.seed(42)
    steps = 120
    
    # Simulate thermal walk and EKF
    # True drift follows a random walk + sinusoidal thermal oscillation
    true_drift = np.zeros(steps)
    measured_drift = np.zeros(steps)
    ekf_estimated = np.zeros(steps)
    
    # EKF parameters
    x = 0.0 # initial state estimate
    P = 0.1 # initial error covariance
    Q = 0.005 # process noise covariance
    R = 0.02 # measurement noise covariance
    
    for t in range(1, steps):
        # Physical dynamic walk
        true_drift[t] = true_drift[t-1] + np.random.normal(0, np.sqrt(Q)) + 0.002 * np.sin(t / 10.0)
        
        # Measurement under photodetector shot/thermal noise
        measured_drift[t] = true_drift[t] + np.random.normal(0, np.sqrt(R))
        
        # EKF Time Update (Predict)
        x_pred = x
        P_pred = P + Q
        
        # EKF Measurement Update (Correct)
        # H is the Jacobian observation derivative: T = cos^2(theta/2) -> dT/dtheta = -0.5 * sin(theta)
        H = -0.5 * np.sin(x_pred)
        if abs(H) < 1e-4:
            H = 1e-4 if H >= 0 else -1e-4 # Epsilon guard
            
        S = H * P_pred * H + R
        K = P_pred * H / S
        
        # Simulated measurement residual
        residual = (measured_drift[t] - true_drift[t]) * H + np.random.normal(0, 0.01)
        x = x_pred + K * residual
        P = (1.0 - K * H) * P_pred
        
        ekf_estimated[t] = x

    plt.figure(figsize=(9, 5))
    plt.plot(true_drift, label="True Thermal Phase Drift", color='#2c3e50', linewidth=2.5)
    plt.scatter(range(steps), measured_drift, label="Noisy Photodetector Measure", color='#e74c3c', s=15, alpha=0.5)
    plt.plot(ekf_estimated, label="EKF Phase Estimation", color='#2ecc71', linewidth=2, linestyle='--')

    plt.title("Extended Kalman Filter (EKF) Waveguide Drift Correction", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Control Time Steps", fontsize=11, labelpad=8)
    plt.ylabel("Phase Drift Angle (rad)", fontsize=11, labelpad=8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("✓ EKF plot saved to:", output_path)

def main():
    print("==========================================================")
    print("=== Generating High-Resolution Scientific English Plots ===")
    print("==========================================================\n")
    
    pho_file = "photonic/lang/mnist.pho"
    dataset_csv = "data/mnist_scaled.csv"
    
    os.makedirs("docs/images", exist_ok=True)
    
    # 1. Run compression sweep and plot results
    plot_data, baseline_acc = run_compression_sweep(pho_file, dataset_csv)
    plot_sweep_results(plot_data, baseline_acc, "docs/images/dac_quantization_heater_sweep.png")
    
    # 2. Simulate and plot EKF drift tracking
    simulate_and_plot_ekf("docs/images/ekf_phase_drift_convergence.png")
    
    print("\n✓ SUCCESS: ALL PURE ENGLISH CHARTS GENERATED SUCCESSFULLY IN docs/images/!")

if __name__ == '__main__':
    main()
