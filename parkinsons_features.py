import parselmouth
import numpy as np
import pandas as pd
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

def calculate_rpde(signal, m=5, r=0.2):
    """Calculate Recurrence Period Density Entropy (RPDE)"""
    try:
        # Simplified RPDE calculation
        N = len(signal)
        if N < 100:
            return np.nan
            
        # Create time-delayed embedding
        tau = 1
        embedded = np.array([signal[i:i+m] for i in range(N-m+1)])
        
        # Calculate recurrence matrix
        distances = np.sqrt(np.sum((embedded[:, None] - embedded) ** 2, axis=2))
        recurrence_matrix = distances < r * np.std(signal)
        
        # Calculate periods
        periods = []
        for i in range(len(recurrence_matrix)):
            recurrent_points = np.where(recurrence_matrix[i])[0]
            if len(recurrent_points) > 1:
                periods.extend(np.diff(recurrent_points))
        
        if len(periods) == 0:
            return np.nan
            
        # Calculate entropy of period distribution
        hist, _ = np.histogram(periods, bins=50)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return entropy(prob)
    except:
        return np.nan

def calculate_dfa(signal, min_box_size=4, max_box_size=None):
    """Calculate Detrended Fluctuation Analysis (DFA)"""
    try:
        if max_box_size is None:
            max_box_size = len(signal) // 4
            
        signal = np.array(signal)
        N = len(signal)
        
        if N < 16:
            return np.nan
            
        # Integrate the signal
        y = np.cumsum(signal - np.mean(signal))
        
        # Calculate fluctuation for different box sizes
        box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), 10).astype(int)
        box_sizes = np.unique(box_sizes)
        
        fluctuations = []
        for box_size in box_sizes:
            if box_size >= N:
                continue
                
            # Divide signal into boxes
            n_boxes = N // box_size
            boxes = y[:n_boxes * box_size].reshape(n_boxes, box_size)
            
            # Detrend each box
            trends = []
            for box in boxes:
                x = np.arange(len(box))
                trend = np.polyfit(x, box, 1)
                trends.append(np.polyval(trend, x))
            
            trends = np.array(trends).flatten()
            detrended = y[:len(trends)] - trends
            
            # Calculate fluctuation
            fluctuation = np.sqrt(np.mean(detrended ** 2))
            fluctuations.append(fluctuation)
        
        if len(fluctuations) < 3:
            return np.nan
            
        # Fit line to log-log plot
        log_box_sizes = np.log10(box_sizes[:len(fluctuations)])
        log_fluctuations = np.log10(fluctuations)
        
        slope, _ = np.polyfit(log_box_sizes, log_fluctuations, 1)
        return slope
    except:
        return np.nan

def calculate_ppe(f0_values):
    """Calculate Pitch Period Entropy (PPE)"""
    try:
        f0_values = f0_values[~np.isnan(f0_values)]
        if len(f0_values) < 10:
            return np.nan
            
        # Calculate pitch periods
        periods = 1.0 / f0_values
        
        # Calculate relative differences
        period_diffs = np.diff(periods)
        relative_diffs = np.abs(period_diffs) / periods[:-1]
        
        # Calculate entropy
        hist, _ = np.histogram(relative_diffs, bins=50)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return entropy(prob)
    except:
        return np.nan

def extract_parkinsons_features(file_path):
    """Extract Parkinson's disease specific voice features"""
    try:
        # Load audio
        snd = parselmouth.Sound(file_path)
        
        # Extract pitch
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]  # Remove unvoiced frames
        
        # Extract point process for jitter/shimmer calculations
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        
        features = {}
        
        # Check if we have enough voiced segments
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        if num_points < 3:
            print(f"Warning: Only {num_points} voiced segments found. Filling with NaN values.")
            # Fill all features with NaN
            feature_names = [
                "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
                "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
                "NHR", "HNR", "RPDE", "DFA", "PPE"
            ]
            return {name: np.nan for name in feature_names}
        
        # Jitter features (in percentage and absolute)
        try:
            jitter_local = parselmouth.praat.call([point_process, pitch], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter(%)"] = jitter_local * 100  # Convert to percentage
            
            jitter_absolute = parselmouth.praat.call([point_process, pitch], "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter(Abs)"] = jitter_absolute
            
            jitter_rap = parselmouth.praat.call([point_process, pitch], "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter:RAP"] = jitter_rap
            
            jitter_ppq5 = parselmouth.praat.call([point_process, pitch], "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter:PPQ5"] = jitter_ppq5
            
            jitter_ddp = parselmouth.praat.call([point_process, pitch], "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter:DDP"] = jitter_ddp
        except Exception as e:
            print(f"Error calculating jitter: {e}")
            features.update({
                "Jitter(%)": np.nan, "Jitter(Abs)": np.nan, "Jitter:RAP": np.nan,
                "Jitter:PPQ5": np.nan, "Jitter:DDP": np.nan
            })
        
        # Shimmer features
        try:
            shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer"] = shimmer_local
            
            shimmer_db = parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer(dB)"] = shimmer_db
            
            shimmer_apq3 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer:APQ3"] = shimmer_apq3
            
            shimmer_apq5 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer:APQ5"] = shimmer_apq5
            
            shimmer_apq11 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer:APQ11"] = shimmer_apq11
            
            shimmer_dda = parselmouth.praat.call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features["Shimmer:DDA"] = shimmer_dda
        except Exception as e:
            print(f"Error calculating shimmer: {e}")
            features.update({
                "Shimmer": np.nan, "Shimmer(dB)": np.nan, "Shimmer:APQ3": np.nan,
                "Shimmer:APQ5": np.nan, "Shimmer:APQ11": np.nan, "Shimmer:DDA": np.nan
            })
        
        # Harmonics-to-Noise Ratio and Noise-to-Harmonics Ratio
        try:
            harmonicity = snd.to_harmonicity_cc()
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
            features["HNR"] = hnr
            features["NHR"] = 1 / (10 ** (hnr / 10)) if hnr > -np.inf else np.nan
        except Exception as e:
            print(f"Error calculating HNR/NHR: {e}")
            features["HNR"] = np.nan
            features["NHR"] = np.nan
        
        # Nonlinear dynamics features
        try:
            if len(f0_values) > 50:
                features["RPDE"] = calculate_rpde(f0_values)
                features["DFA"] = calculate_dfa(f0_values)
                features["PPE"] = calculate_ppe(f0_values)
            else:
                features["RPDE"] = np.nan
                features["DFA"] = np.nan
                features["PPE"] = np.nan
        except Exception as e:
            print(f"Error calculating nonlinear features: {e}")
            features["RPDE"] = np.nan
            features["DFA"] = np.nan
            features["PPE"] = np.nan
        
        return features
        
    except Exception as e:
        print(f"Error processing audio file: {e}")
        # Return all NaN values
        feature_names = [
            "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
            "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
            "NHR", "HNR", "RPDE", "DFA", "PPE"
        ]
        return {name: np.nan for name in feature_names}

# Extract features
print("Extracting Parkinson's disease specific voice features...")
features = extract_parkinsons_features("WhatsApp Audio 2025-07-03 at 17.11.51_77c6b56d.wav")

# Display results
print("\nExtracted Features:")
print("=" * 50)
for feature_name, value in features.items():
    if np.isnan(value):
        print(f"{feature_name}: Not Available")
    else:
        print(f"{feature_name}: {value:.6f}")

# Save to CSV
df = pd.DataFrame([features])
df.to_csv("parkinsons_features.csv", index=False)
print(f"\nFeatures saved to 'parkinsons_features.csv'")

# Create a summary
print("\nFeature Descriptions:")
print("=" * 50)
descriptions = {
    "Jitter(%)": "Variation in fundamental frequency (percentage)",
    "Jitter(Abs)": "Absolute variation in fundamental frequency",
    "Jitter:RAP": "Relative Average Perturbation of jitter",
    "Jitter:PPQ5": "Five-point Period Perturbation Quotient",
    "Jitter:DDP": "Average absolute difference of differences between cycles",
    "Shimmer": "Variation in amplitude",
    "Shimmer(dB)": "Variation in amplitude (decibels)",
    "Shimmer:APQ3": "Three-point Amplitude Perturbation Quotient",
    "Shimmer:APQ5": "Five-point Amplitude Perturbation Quotient",
    "Shimmer:APQ11": "Eleven-point Amplitude Perturbation Quotient", 
    "Shimmer:DDA": "Average absolute difference of differences between amplitudes",
    "NHR": "Noise-to-Harmonics Ratio",
    "HNR": "Harmonics-to-Noise Ratio",
    "RPDE": "Recurrence Period Density Entropy (nonlinear complexity)",
    "DFA": "Detrended Fluctuation Analysis (fractal scaling)",
    "PPE": "Pitch Period Entropy (fundamental frequency variation)"
}

for feature, desc in descriptions.items():
    print(f"{feature}: {desc}")
