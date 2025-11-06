
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Define a constant for the number of data points to display
WINDOW_SIZE = 500

def charts(eeg_data):
    st.header("Real-Time EEG Data")
    chart_placeholder = st.empty()

    if eeg_data:
        # Use a sliding window to display only the most recent data
        recent_data = eeg_data[-WINDOW_SIZE:]

        # Create a DataFrame from the recent data
        # Assuming the data format is a list of lists/tuples [channel1, channel2, ...]
        df = pd.DataFrame(recent_data, columns=[f"Channel {i+1}" for i in range(len(recent_data[0]))])

        fig = go.Figure()
        for column in df.columns:
            # Using 'lines' mode for a continuous trace
            fig.add_trace(go.Scatter(y=df[column], name=column, mode='lines'))

        fig.update_layout(
            title="Live EEG Channels",
            xaxis_title="Time (Samples)",
            yaxis_title="Voltage (uV)",
            height=400,
            legend_title="Channels"
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        # Display a placeholder chart when no data is available
        st.info("Start the data stream to see live EEG.")
        placeholder_data = np.random.randn(100, 5)
        df = pd.DataFrame(placeholder_data, columns=[f"Channel {i+1}" for i in range(5)])

        fig = go.Figure()
        for column in df.columns:
            fig.add_trace(go.Scatter(y=df[column], name=column, mode='lines'))

        fig.update_layout(
            title="EEG Channels (Placeholder)",
            xaxis_title="Time (Samples)",
            yaxis_title="Voltage (uV)",
            height=400
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True)
