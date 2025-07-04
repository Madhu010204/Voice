import streamlit as st

# Simple test to check if Streamlit works
st.title("🎤 Parkinson's Voice Analysis - Test")
st.write("If you can see this, Streamlit is working correctly!")

# Test the import of required packages
try:
    import parselmouth
    st.success("✅ Parselmouth imported successfully")
except ImportError:
    st.error("❌ Parselmouth not found")

try:
    import pandas as pd
    st.success("✅ Pandas imported successfully")
except ImportError:
    st.error("❌ Pandas not found")

try:
    import plotly.express as px
    st.success("✅ Plotly imported successfully")
except ImportError:
    st.error("❌ Plotly not found")

st.write("All dependencies are ready! You can now use the main application.")
