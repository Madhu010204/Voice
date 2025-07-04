import parselmouth
import os


def extract_voice_features(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        
        # Check voiced pitch pulses
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        if num_points < 3:
            print(f"⚠️ Warning: Only {num_points} pitch pulses detected. Jitter/Shimmer may be unreliable.")
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

    # Extract Jitter features
    try:
        jitter_local = parselmouth.praat.call([point_process, pitch], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_absolute = parselmouth.praat.call([point_process, pitch], "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = parselmouth.praat.call([point_process, pitch], "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ppq5 = parselmouth.praat.call([point_process, pitch], "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_ddp = parselmouth.praat.call([point_process, pitch], "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception as e:
        print(f"❌ Error extracting jitter: {e}")
        jitter_local = jitter_absolute = jitter_rap = jitter_ppq5 = jitter_ddp = float('nan')

    # Extract Shimmer features
    try:
        shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_db = parselmouth.praat.call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq5 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq11 = parselmouth.praat.call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_dda = parselmouth.praat.call([snd, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception as e:
        print(f"❌ Error extracting shimmer: {e}")
        shimmer_local = shimmer_db = shimmer_apq3 = shimmer_apq5 = shimmer_apq11 = shimmer_dda = float('nan')

    # Extract HNR and NHR
    try:
        harmonicity = snd.to_harmonicity_cc()
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        nhr = 1 / (10 ** (hnr / 10)) if hnr != 0 else float('inf')
    except Exception as e:
        print(f"❌ Error calculating HNR/NHR: {e}")
        hnr = nhr = float('nan')

    return {
        "Jitter(%)": jitter_local,
        "Jitter(Abs)": jitter_absolute,
        "Jitter:RAP": jitter_rap,
        "Jitter:PPQ5": jitter_ppq5,
        "Jitter:DDP": jitter_ddp,
        "Shimmer": shimmer_local,
        "Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": shimmer_apq3,
        "Shimmer:APQ5": shimmer_apq5,
        "Shimmer:APQ11": shimmer_apq11,
        "Shimmer:DDA": shimmer_dda,
        "HNR": hnr,
        "NHR": nhr
    }

# Example
features = extract_voice_features("WhatsApp Audio 2025-07-03 at 17.11.51_77c6b56d.wav")
print(features)