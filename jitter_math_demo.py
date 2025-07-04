import numpy as np
import matplotlib.pyplot as plt

def demonstrate_jitter_calculations():
    """
    Demonstrate mathematical jitter calculations with example data
    """
    
    # Example F0 values (Hz) - simulating voice with some variation
    np.random.seed(42)
    base_f0 = 150  # 150 Hz base frequency
    n_samples = 50
    
    # Add some realistic variation to simulate voice
    f0_values = base_f0 + np.random.normal(0, 5, n_samples)  # Small variations
    f0_values = np.abs(f0_values)  # Ensure positive frequencies
    
    print("MATHEMATICAL JITTER CALCULATION DEMONSTRATION")
    print("=" * 55)
    print(f"Example F0 values: {f0_values[:10]:.2f} Hz (showing first 10)")
    print(f"Total samples: {len(f0_values)}")
    print()
    
    # Convert to periods
    periods = 1.0 / f0_values
    N = len(periods)
    mean_period = np.mean(periods)
    
    print("STEP-BY-STEP CALCULATIONS:")
    print("-" * 30)
    
    # 1. Jitter (Local)
    print("1. JITTER LOCAL:")
    period_diffs = np.abs(np.diff(periods))
    jitter_local = np.mean(period_diffs) / mean_period
    jitter_percent = jitter_local * 100
    jitter_absolute = np.mean(period_diffs)
    
    print(f"   Formula: Σ|T(i+1) - T(i)| / (N-1) / mean(T)")
    print(f"   Period differences: {period_diffs[:5]:.6f} ... (first 5)")
    print(f"   Mean period difference: {np.mean(period_diffs):.6f} seconds")
    print(f"   Mean period: {mean_period:.6f} seconds")
    print(f"   Jitter(%): {jitter_percent:.4f}%")
    print(f"   Jitter(Abs): {jitter_absolute:.6f} seconds")
    print()
    
    # 2. RAP (Relative Average Perturbation)
    print("2. RAP (3-point smoothing):")
    if N >= 3:
        rap_sum = 0
        for i in range(1, N-1):
            local_mean = (periods[i-1] + periods[i] + periods[i+1]) / 3
            rap_sum += abs(periods[i] - local_mean)
        jitter_rap = (rap_sum / (N-2)) / mean_period
        
        print(f"   Formula: Σ|T(i) - (T(i-1)+T(i)+T(i+1))/3| / (N-2) / mean(T)")
        print(f"   Example for i=1: |{periods[1]:.6f} - {(periods[0]+periods[1]+periods[2])/3:.6f}| = {abs(periods[1] - (periods[0]+periods[1]+periods[2])/3):.6f}")
        print(f"   RAP: {jitter_rap:.6f}")
    else:
        jitter_rap = np.nan
        print("   Insufficient data (need ≥3 periods)")
    print()
    
    # 3. PPQ5 (5-point Period Perturbation Quotient)
    print("3. PPQ5 (5-point smoothing):")
    if N >= 5:
        ppq5_sum = 0
        for i in range(2, N-2):
            local_mean = np.mean(periods[i-2:i+3])
            ppq5_sum += abs(periods[i] - local_mean)
        jitter_ppq5 = (ppq5_sum / (N-4)) / mean_period
        
        print(f"   Formula: Σ|T(i) - mean(T(i-2:i+2))| / (N-4) / mean(T)")
        print(f"   Example for i=2: |{periods[2]:.6f} - {np.mean(periods[0:5]):.6f}| = {abs(periods[2] - np.mean(periods[0:5])):.6f}")
        print(f"   PPQ5: {jitter_ppq5:.6f}")
    else:
        jitter_ppq5 = np.nan
        print("   Insufficient data (need ≥5 periods)")
    print()
    
    # 4. DDP (Difference of Differences of Periods)
    print("4. DDP (Difference of differences):")
    if N >= 3:
        first_diffs = np.diff(periods)
        second_diffs = np.diff(first_diffs)
        jitter_ddp = np.mean(np.abs(second_diffs)) / mean_period
        
        print(f"   Formula: Σ|(T(i+1)-T(i)) - (T(i)-T(i-1))| / (N-2) / mean(T)")
        print(f"   First differences: {first_diffs[:5]:.6f} ... (first 5)")
        print(f"   Second differences: {second_diffs[:5]:.6f} ... (first 5)")
        print(f"   DDP: {jitter_ddp:.6f}")
    else:
        jitter_ddp = np.nan
        print("   Insufficient data (need ≥3 periods)")
    print()
    
    # Summary
    print("FINAL RESULTS:")
    print("=" * 20)
    print(f"Jitter(%):    {jitter_percent:.4f}%")
    print(f"Jitter(Abs):  {jitter_absolute:.6f} seconds")
    print(f"Jitter:RAP:   {jitter_rap:.6f}")
    print(f"Jitter:PPQ5:  {jitter_ppq5:.6f}")
    print(f"Jitter:DDP:   {jitter_ddp:.6f}")
    print()
    
    # Clinical interpretation
    print("CLINICAL INTERPRETATION:")
    print("-" * 25)
    normal_jitter = 0.5  # Normal jitter is typically < 0.5%
    if jitter_percent < normal_jitter:
        print(f"✅ Jitter({jitter_percent:.2f}%) is within normal range (<{normal_jitter}%)")
    else:
        print(f"⚠️  Jitter({jitter_percent:.2f}%) exceeds normal range (>{normal_jitter}%)")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: F0 values over time
    plt.subplot(2, 2, 1)
    plt.plot(f0_values, 'b-', linewidth=2)
    plt.title('F0 Values Over Time')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Sample')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Periods
    plt.subplot(2, 2, 2)
    plt.plot(periods, 'g-', linewidth=2)
    plt.title('Periods (T = 1/F0)')
    plt.ylabel('Period (seconds)')
    plt.xlabel('Sample')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Period differences
    plt.subplot(2, 2, 3)
    plt.plot(period_diffs, 'r-', linewidth=2)
    plt.title('Period Differences |T(i+1) - T(i)|')
    plt.ylabel('Difference (seconds)')
    plt.xlabel('Sample')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative jitter measures
    plt.subplot(2, 2, 4)
    measures = ['Jitter(%)', 'RAP', 'PPQ5', 'DDP']
    values = [jitter_percent, jitter_rap*100, jitter_ppq5*100, jitter_ddp*100]  # Scale for visibility
    colors = ['blue', 'green', 'orange', 'red']
    
    bars = plt.bar(measures, values, color=colors, alpha=0.7)
    plt.title('Jitter Measures Comparison')
    plt.ylabel('Value (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'jitter_percent': jitter_percent,
        'jitter_absolute': jitter_absolute,
        'jitter_rap': jitter_rap,
        'jitter_ppq5': jitter_ppq5,
        'jitter_ddp': jitter_ddp,
        'f0_values': f0_values,
        'periods': periods
    }

if __name__ == "__main__":
    results = demonstrate_jitter_calculations()
