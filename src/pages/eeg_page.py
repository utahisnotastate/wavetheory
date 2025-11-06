
import streamlit as st
from src.state.use_recording import use_recording
from src.ui.eeg.device_card import device_card
from src.ui.eeg.charts import charts
from src.ui.eeg.recorder import recorder
from src.ui.eeg.annotator import annotator

def eeg_page():
    st.title("🧠 Wish Machine")

    recording_state = use_recording()
    recording_state["process_eeg_data"]()

    col1, col2 = st.columns([1, 2])

    with col1:
        device_card()
        st.divider()
        recorder()
        st.divider()
        annotator()

    with col2:
        charts(recording_state["eeg_data"])
