import pandas as pd
import numpy as np

# Read the features
df = pd.read_csv("parkinsons_features.csv")

print("PARKINSON'S DISEASE VOICE FEATURES ANALYSIS")
print("=" * 55)
print(f"Audio File: WhatsApp Audio 2025-07-03 at 17.11.51_77c6b56d.wav")
print("=" * 55)

# Define normal ranges for comparison (typical healthy values)
normal_ranges = {
    "Jitter(%)": (0.1, 0.5),
    "Jitter(Abs)": (0.00002, 0.00005),
    "Jitter:RAP": (0.1, 0.5),
    "Jitter:PPQ5": (0.1, 0.5),
    "Jitter:DDP": (0.3, 1.5),
    "Shimmer": (0.02, 0.05),
    "Shimmer(dB)": (0.2, 0.5),
    "Shimmer:APQ3": (0.01, 0.04),
    "Shimmer:APQ5": (0.02, 0.05),
    "Shimmer:APQ11": (0.03, 0.08),
    "Shimmer:DDA": (0.03, 0.12),
    "HNR": (20, 30),  # Higher is better
    "NHR": (0.01, 0.05),  # Lower is better
    "RPDE": (0.4, 0.7),
    "DFA": (0.5, 0.9),
    "PPE": (0.1, 0.3)
}

feature_descriptions = {
    "Jitter(%)": "Fundamental frequency variation (%)",
    "Jitter(Abs)": "Absolute frequency variation",
    "Jitter:RAP": "Relative Average Perturbation",
    "Jitter:PPQ5": "5-point Period Perturbation Quotient",
    "Jitter:DDP": "Average absolute difference of periods",
    "Shimmer": "Amplitude variation",
    "Shimmer(dB)": "Amplitude variation (dB)",
    "Shimmer:APQ3": "3-point Amplitude Perturbation Quotient",
    "Shimmer:APQ5": "5-point Amplitude Perturbation Quotient",
    "Shimmer:APQ11": "11-point Amplitude Perturbation Quotient",
    "Shimmer:DDA": "Average absolute amplitude differences",
    "HNR": "Harmonics-to-Noise Ratio (dB)",
    "NHR": "Noise-to-Harmonics Ratio",
    "RPDE": "Recurrence Period Density Entropy",
    "DFA": "Detrended Fluctuation Analysis",
    "PPE": "Pitch Period Entropy"
}

# Print each feature with analysis
for feature in df.columns:
    value = df[feature].iloc[0]
    
    if pd.isna(value):
        status = "❌ NOT AVAILABLE"
        interpretation = "Insufficient voiced segments for calculation"
    else:
        # Check if value is within normal range
        if feature in normal_ranges:
            min_val, max_val = normal_ranges[feature]
            if min_val <= value <= max_val:
                status = "✅ NORMAL"
                interpretation = "Within typical healthy range"
            elif feature in ["HNR"] and value > max_val:
                status = "✅ EXCELLENT"
                interpretation = "Better than typical (very good voice quality)"
            elif feature in ["NHR"] and value < min_val:
                status = "✅ EXCELLENT"
                interpretation = "Better than typical (very low noise)"
            else:
                status = "⚠️  ABNORMAL"
                if feature.startswith("Jitter") or feature.startswith("Shimmer") or feature in ["NHR", "RPDE", "DFA", "PPE"]:
                    interpretation = "Higher than typical (may indicate voice disorder)"
                elif feature == "HNR":
                    interpretation = "Lower than typical (may indicate voice disorder)"
                else:
                    interpretation = "Outside typical range"
        else:
            status = "📊 MEASURED"
            interpretation = "Reference ranges not established"
    
    desc = feature_descriptions.get(feature, "Voice feature")
    
    print(f"\n{feature}:")
    print(f"  Description: {desc}")
    if not pd.isna(value):
        print(f"  Value: {value:.6f}")
    else:
        print(f"  Value: Not Available")
    print(f"  Status: {status}")
    print(f"  Interpretation: {interpretation}")

# Overall assessment
print("\n" + "=" * 55)
print("OVERALL VOICE ANALYSIS SUMMARY")
print("=" * 55)

# Count available features
available_features = df.notna().sum().sum()
total_features = len(df.columns)

print(f"Features Extracted: {available_features}/{total_features}")

# Analyze key indicators
key_indicators = {}
if not pd.isna(df["HNR"].iloc[0]):
    hnr_val = df["HNR"].iloc[0]
    key_indicators["Voice Quality (HNR)"] = "Good" if hnr_val > 15 else "Concerning"

if not pd.isna(df["Shimmer"].iloc[0]):
    shimmer_val = df["Shimmer"].iloc[0]
    key_indicators["Amplitude Stability"] = "Good" if shimmer_val < 0.1 else "Concerning"

if available_features >= 10:
    overall_status = "✅ COMPREHENSIVE ANALYSIS COMPLETED"
else:
    overall_status = "⚠️  LIMITED ANALYSIS - NEED MORE VOICED SEGMENTS"

print(f"Analysis Status: {overall_status}")

if key_indicators:
    print("\nKey Voice Quality Indicators:")
    for indicator, status in key_indicators.items():
        print(f"  • {indicator}: {status}")

print(f"\nNote: This analysis is for research/educational purposes only.")
print(f"      Clinical diagnosis requires professional medical evaluation.")
