import streamlit as st
import parselmouth
import numpy as np
import pandas as pd
from scipy.stats import entropy
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
import tempfile

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Parkinson's Voice Analysis",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .normal-range {
        color: #28a745;
        font-weight: bold;
    }
    .abnormal-range {
        color: #dc3545;
        font-weight: bold;
    }
    .warning-range {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def calculate_rpde(signal, m=5, r=0.2):
    """Calculate Recurrence Period Density Entropy (RPDE)"""
    try:
        N = len(signal)
        if N < 100:
            return np.nan
            
        tau = 1
        embedded = np.array([signal[i:i+m] for i in range(N-m+1)])
        
        distances = np.sqrt(np.sum((embedded[:, None] - embedded) ** 2, axis=2))
        recurrence_matrix = distances < r * np.std(signal)
        
        periods = []
        for i in range(len(recurrence_matrix)):
            recurrent_points = np.where(recurrence_matrix[i])[0]
            if len(recurrent_points) > 1:
                periods.extend(np.diff(recurrent_points))
        
        if len(periods) == 0:
            return np.nan
            
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
            
        y = np.cumsum(signal - np.mean(signal))
        
        box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), 10).astype(int)
        box_sizes = np.unique(box_sizes)
        
        fluctuations = []
        for box_size in box_sizes:
            if box_size >= N:
                continue
                
            n_boxes = N // box_size
            boxes = y[:n_boxes * box_size].reshape(n_boxes, box_size)
            
            trends = []
            for box in boxes:
                x = np.arange(len(box))
                trend = np.polyfit(x, box, 1)
                trends.append(np.polyval(trend, x))
            
            trends = np.array(trends).flatten()
            detrended = y[:len(trends)] - trends
            
            fluctuation = np.sqrt(np.mean(detrended ** 2))
            fluctuations.append(fluctuation)
        
        if len(fluctuations) < 3:
            return np.nan
            
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
            
        periods = 1.0 / f0_values
        
        period_diffs = np.diff(periods)
        relative_diffs = np.abs(period_diffs) / periods[:-1]
        
        hist, _ = np.histogram(relative_diffs, bins=50)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return entropy(prob)
    except:
        return np.nan

def calculate_jitter_mathematical(f0_values):
    """Calculate jitter features using mathematical formulas"""
    try:
        # Remove NaN values and convert to periods
        f0_values = f0_values[~np.isnan(f0_values)]
        if len(f0_values) < 10:
            return {
                "Jitter(%)": np.nan,
                "Jitter(Abs)": np.nan,
                "Jitter:RAP": np.nan,
                "Jitter:PPQ5": np.nan,
                "Jitter:DDP": np.nan
            }
        
        # Convert frequencies to periods (T = 1/F0)
        periods = 1.0 / f0_values
        N = len(periods)
        
        # 1. Jitter (Local) - average absolute difference between consecutive periods
        period_diffs = np.abs(np.diff(periods))
        mean_period = np.mean(periods)
        
        jitter_local = np.mean(period_diffs) / mean_period
        jitter_percent = jitter_local * 100  # Convert to percentage
        jitter_absolute = np.mean(period_diffs)  # Absolute jitter in seconds
        
        # 2. RAP (Relative Average Perturbation) - 3-point smoothing
        if N >= 3:
            rap_sum = 0
            for i in range(1, N-1):
                # Compare each period with average of itself and neighbors
                local_mean = (periods[i-1] + periods[i] + periods[i+1]) / 3
                rap_sum += abs(periods[i] - local_mean)
            jitter_rap = (rap_sum / (N-2)) / mean_period
        else:
            jitter_rap = np.nan
        
        # 3. PPQ5 (5-point Period Perturbation Quotient)
        if N >= 5:
            ppq5_sum = 0
            for i in range(2, N-2):
                # Compare each period with average of 5-point window
                local_mean = np.mean(periods[i-2:i+3])
                ppq5_sum += abs(periods[i] - local_mean)
            jitter_ppq5 = (ppq5_sum / (N-4)) / mean_period
        else:
            jitter_ppq5 = np.nan
        
        # 4. DDP (Difference of Differences of Periods)
        if N >= 3:
            # Calculate first differences
            first_diffs = np.diff(periods)
            # Calculate second differences (differences of differences)
            second_diffs = np.diff(first_diffs)
            # DDP is average absolute second difference
            jitter_ddp = np.mean(np.abs(second_diffs)) / mean_period
        else:
            jitter_ddp = np.nan
        
        return {
            "Jitter(%)": jitter_percent,
            "Jitter(Abs)": jitter_absolute,
            "Jitter:RAP": jitter_rap,
            "Jitter:PPQ5": jitter_ppq5,
            "Jitter:DDP": jitter_ddp
        }
        
    except Exception as e:
        return {
            "Jitter(%)": np.nan,
            "Jitter(Abs)": np.nan,
            "Jitter:RAP": np.nan,
            "Jitter:PPQ5": np.nan,
            "Jitter:DDP": np.nan
        }

def extract_parkinsons_features(audio_file):
    """Extract Parkinson's disease specific voice features"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            temp_path = tmp_file.name
        
        # Load audio
        snd = parselmouth.Sound(temp_path)
        
        # Extract pitch
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]
        
        # Extract point process
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        
        features = {}
        warnings_list = []
        
        # Check voiced segments
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        if num_points < 3:
            warnings_list.append(f"Only {num_points} voiced segments found. Some features may be unreliable.")
        
        # Jitter features - try both Praat and mathematical methods
        try:
            # First try Praat method
            jitter_local = parselmouth.praat.call([point_process, pitch], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter(%)"] = jitter_local * 100
            
            jitter_absolute = parselmouth.praat.call([point_process, pitch], "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter(Abs)"] = jitter_absolute
            
            jitter_rap = parselmouth.praat.call([point_process, pitch], "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter:RAP"] = jitter_rap
            
            jitter_ppq5 = parselmouth.praat.call([point_process, pitch], "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter:PPQ5"] = jitter_ppq5
            
            jitter_ddp = parselmouth.praat.call([point_process, pitch], "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            features["Jitter:DDP"] = jitter_ddp
            
            # Add method indicator
            features["Jitter_Method"] = "Praat"
            
        except Exception as e:
            warnings_list.append(f"Praat jitter calculation failed: {str(e)}")
            warnings_list.append("Using mathematical jitter calculation as fallback...")
            
            # Fallback to mathematical method
            math_jitter = calculate_jitter_mathematical(f0_values)
            features.update(math_jitter)
            features["Jitter_Method"] = "Mathematical"
        
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
            warnings_list.append(f"Could not calculate shimmer features: {str(e)}")
            features.update({
                "Shimmer": np.nan, "Shimmer(dB)": np.nan, "Shimmer:APQ3": np.nan,
                "Shimmer:APQ5": np.nan, "Shimmer:APQ11": np.nan, "Shimmer:DDA": np.nan
            })
        
        # HNR/NHR
        try:
            harmonicity = snd.to_harmonicity_cc()
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
            features["HNR"] = hnr
            features["NHR"] = 1 / (10 ** (hnr / 10)) if hnr > -np.inf else np.nan
        except Exception as e:
            warnings_list.append(f"Could not calculate HNR/NHR: {str(e)}")
            features["HNR"] = np.nan
            features["NHR"] = np.nan
        
        # Nonlinear features
        try:
            if len(f0_values) > 50:
                features["RPDE"] = calculate_rpde(f0_values)
                features["DFA"] = calculate_dfa(f0_values)
                features["PPE"] = calculate_ppe(f0_values)
            else:
                features["RPDE"] = np.nan
                features["DFA"] = np.nan
                features["PPE"] = np.nan
                warnings_list.append("Insufficient data for nonlinear analysis features")
        except Exception as e:
            warnings_list.append(f"Could not calculate nonlinear features: {str(e)}")
            features["RPDE"] = np.nan
            features["DFA"] = np.nan
            features["PPE"] = np.nan
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return features, warnings_list, f0_values, snd
        
    except Exception as e:
        return None, [f"Error processing audio file: {str(e)}"], None, None

def get_feature_status(feature_name, value):
    """Determine if feature value is normal, abnormal, or excellent"""
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
        "HNR": (20, 30),
        "NHR": (0.01, 0.05),
        "RPDE": (0.4, 0.7),
        "DFA": (0.5, 0.9),
        "PPE": (0.1, 0.3)
    }
    
    if pd.isna(value):
        return "N/A", "gray"
    
    if feature_name not in normal_ranges:
        return "Measured", "blue"
    
    min_val, max_val = normal_ranges[feature_name]
    
    if min_val <= value <= max_val:
        return "Normal", "green"
    elif feature_name == "HNR" and value > max_val:
        return "Excellent", "darkgreen"
    elif feature_name == "NHR" and value < min_val:
        return "Excellent", "darkgreen"
    else:
        return "Abnormal", "red"

def create_radar_chart(features):
    """Create a radar chart for voice features"""
    # Select key features for radar chart
    radar_features = ["Jitter(%)", "Shimmer", "HNR", "RPDE", "DFA", "PPE"]
    
    values = []
    labels = []
    
    for feature in radar_features:
        if feature in features and not pd.isna(features[feature]):
            # Normalize values for better visualization
            if feature == "Jitter(%)":
                normalized = min(features[feature] / 2.0, 1.0)  # Cap at 2%
            elif feature == "Shimmer":
                normalized = min(features[feature] / 0.2, 1.0)  # Cap at 0.2
            elif feature == "HNR":
                normalized = max(0, min(features[feature] / 30.0, 1.0))  # Normalize to 30dB max
            else:
                normalized = min(features[feature] / 2.0, 1.0)  # General normalization
            
            values.append(normalized)
            labels.append(feature)
    
    if len(values) < 3:
        return None
    
    # Close the radar chart
    values += values[:1]
    labels += labels[:1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Voice Features',
        line_color='rgb(66, 133, 244)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Voice Features Radar Chart (Normalized)"
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎤 Parkinson\'s Disease Voice Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Analysis Options")
    st.sidebar.markdown("Upload an audio file to analyze voice features commonly used in Parkinson's disease detection research.")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload a voice recording for analysis. WAV format is recommended."
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.sidebar.audio(uploaded_file, format='audio/wav')
        
        # Analysis button
        if st.sidebar.button("🔬 Analyze Voice", type="primary"):
            with st.spinner("Analyzing voice features... This may take a moment."):
                features, warnings_list, f0_values, sound = extract_parkinsons_features(uploaded_file)
            
            if features is not None:
                # Main results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.header("📊 Analysis Results")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📈 Feature Overview", "📋 Detailed Results", "📊 Visualizations"])
                    
                    with tab1:
                        # Key metrics
                        metrics_cols = st.columns(4)
                        
                        key_features = ["HNR", "Shimmer", "Jitter(%)", "RPDE"]
                        for i, feature in enumerate(key_features):
                            if feature in features and not pd.isna(features[feature]):
                                status, color = get_feature_status(feature, features[feature])
                                metrics_cols[i].metric(
                                    label=feature,
                                    value=f"{features[feature]:.4f}",
                                    delta=status
                                )
                        
                        # Overall assessment
                        st.subheader("🎯 Overall Assessment")
                        available_features = sum(1 for k, v in features.items() 
                                               if not pd.isna(v) and isinstance(v, (int, float)) and k != "Jitter_Method")
                        total_features = len([k for k in features.keys() if k != "Jitter_Method"])
                        
                        if available_features >= 12:
                            st.success(f"✅ Comprehensive analysis completed ({available_features}/{total_features} features extracted)")
                        elif available_features >= 8:
                            st.warning(f"⚠️ Partial analysis completed ({available_features}/{total_features} features extracted)")
                        else:
                            st.error(f"❌ Limited analysis ({available_features}/{total_features} features extracted)")
                    
                    with tab2:
                        # Detailed feature table
                        st.subheader("📋 Complete Feature Analysis")
                        
                        feature_descriptions = {
                            "Jitter(%)": "Variation in fundamental frequency (%)",
                            "Jitter(Abs)": "Absolute variation in fundamental frequency",
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
                        
                        # Create detailed results table
                        results_data = []
                        for feature_name, value in features.items():
                            # Skip internal method indicators
                            if feature_name == "Jitter_Method":
                                continue
                                
                            status, color = get_feature_status(feature_name, value)
                            
                            # Format value appropriately
                            if pd.isna(value):
                                formatted_value = "N/A"
                            elif isinstance(value, (int, float)):
                                formatted_value = f"{value:.6f}"
                            else:
                                formatted_value = str(value)
                            
                            results_data.append({
                                "Feature": feature_name,
                                "Value": formatted_value,
                                "Status": status,
                                "Description": feature_descriptions.get(feature_name, "Voice feature")
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                    
                    with tab3:
                        # Visualizations
                        st.subheader("📈 Voice Feature Visualizations")
                        
                        # Radar chart
                        radar_fig = create_radar_chart(features)
                        if radar_fig:
                            st.plotly_chart(radar_fig, use_container_width=True)
                        
                        # Feature distribution
                        if f0_values is not None and len(f0_values) > 0:
                            fig_f0 = px.line(
                                x=range(len(f0_values)), 
                                y=f0_values,
                                title="Fundamental Frequency (F0) Over Time",
                                labels={"x": "Time Frames", "y": "F0 (Hz)"}
                            )
                            st.plotly_chart(fig_f0, use_container_width=True)
                        
                        # Feature comparison bars
                        available_features = {k: v for k, v in features.items() 
                                            if not pd.isna(v) and isinstance(v, (int, float)) and k != "Jitter_Method"}
                        if len(available_features) > 0:
                            fig_bar = px.bar(
                                x=list(available_features.keys()),
                                y=list(available_features.values()),
                                title="Extracted Voice Features",
                                labels={"x": "Features", "y": "Values"}
                            )
                            fig_bar.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    st.header("ℹ️ Information")
                    
                    # Warnings
                    if warnings_list:
                        st.subheader("⚠️ Analysis Warnings")
                        for warning in warnings_list:
                            st.warning(warning)
                    
                    # Feature info
                    st.subheader("📖 About the Features")
                    st.markdown("""
                    **Jitter Features**: Measure variation in vocal pitch
                    - Higher values may indicate voice disorders
                    
                    **Shimmer Features**: Measure variation in vocal amplitude  
                    - Higher values may indicate voice disorders
                    
                    **HNR/NHR**: Voice quality measures
                    - HNR: Higher is better (less noise)
                    - NHR: Lower is better (less noise)
                    
                    **Nonlinear Features**: Complex voice dynamics
                    - RPDE, DFA, PPE: Measure voice complexity and stability
                    """)
                    
                    # Mathematical formulas expander
                    with st.expander("🔢 Mathematical Formulas"):
                        st.markdown("""
                        **Jitter Calculations:**
                        
                        Given fundamental frequency values F₀, convert to periods: T = 1/F₀
                        
                        • **Jitter(%)**: `100 × (Σ|Tᵢ₊₁ - Tᵢ|/(N-1)) / (ΣTᵢ/N)`
                        
                        • **Jitter(Abs)**: `Σ|Tᵢ₊₁ - Tᵢ|/(N-1)` (in seconds)
                        
                        • **RAP**: `(Σ|Tᵢ - (Tᵢ₋₁+Tᵢ+Tᵢ₊₁)/3|/(N-2)) / T̄`
                        
                        • **PPQ5**: `(Σ|Tᵢ - T̄₅ᵢ|/(N-4)) / T̄`
                          where T̄₅ᵢ is 5-point local average
                        
                        • **DDP**: `(Σ|(Tᵢ₊₁-Tᵢ) - (Tᵢ-Tᵢ₋₁)|/(N-2)) / T̄`
                        
                        Where:
                        - N = number of periods
                        - T̄ = mean period duration
                        - |x| = absolute value
                        """)
                    
                    # Show calculation method
                    if 'features' in locals() and features is not None and 'Jitter_Method' in features:
                        method = features['Jitter_Method']
                        if method == "Praat":
                            st.info("🔬 Jitter calculated using Praat algorithms")
                        else:
                            st.info("🧮 Jitter calculated using mathematical formulas")
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"voice_analysis_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                
                # Disclaimer
                st.markdown("---")
                st.markdown("""
                **⚠️ Disclaimer**: This tool is for research and educational purposes only. 
                It should not be used for medical diagnosis. Always consult healthcare professionals 
                for medical concerns.
                """)
            
            else:
                st.error("❌ Failed to analyze the audio file. Please check the file format and try again.")
                for warning in warnings_list:
                    st.error(warning)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an audio file using the sidebar to begin analysis.")
        
        # Feature information
        st.header("🔬 About Parkinson's Voice Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Features Analyzed")
            st.markdown("""
            - **Jitter (5 measures)**: Frequency variation
            - **Shimmer (6 measures)**: Amplitude variation  
            - **HNR/NHR**: Voice quality ratios
            - **RPDE**: Recurrence analysis
            - **DFA**: Fractal scaling
            - **PPE**: Pitch entropy
            """)
        
        with col2:
            st.subheader("🎯 Research Applications")
            st.markdown("""
            - Voice disorder detection
            - Parkinson's disease research
            - Speech therapy assessment
            - Voice quality monitoring
            - Clinical voice analysis
            """)

if __name__ == "__main__":
    main()
