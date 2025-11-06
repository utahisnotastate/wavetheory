
import streamlit as st
from src.state.use_recording import use_recording

def recorder():
    recording_state = use_recording()

    # Disable controls if not connected
    is_connected = st.session_state.get('is_connected', False)

    st.header("Data Streaming")
    stream_col1, stream_col2 = st.columns(2)
    with stream_col1:
        if st.button("Start Stream", disabled=not is_connected or recording_state["is_streaming"]):
            client = st.session_state.get('cortex_client')
            token = st.session_state.get('cortex_token')
            # A headset ID would be dynamic in a real app
            headset_id = "Emotiv Insight"
            if client and token:
                recording_state["start_streaming"](client, token, headset_id)
            else:
                st.warning("Connection details not found.")

    with stream_col2:
        if st.button("Stop Stream", disabled=not recording_state["is_streaming"]):
            recording_state["stop_streaming"]()

    st.header("Recording Controls")
    # Recording can only happen if streaming
    can_record = recording_state["is_streaming"]

    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        if st.button("Start Recording", disabled=not can_record or recording_state["recording"]):
            recording_state["start_recording"]()
            st.info("Recording started...")

    with rec_col2:
        if st.button("Stop Recording", disabled=not recording_state["recording"]):
            recording_state["stop_recording"]()
            st.success("Recording stopped. Preparing for upload...")
            upload_eeg_data(recording_state["eeg_data"], recording_state["annotations"])

import json
import datetime
from src.firebase_config import get_firestore_client, get_storage_bucket

def upload_eeg_data(eeg_data, annotations):
    if not eeg_data:
        st.warning("No EEG data to upload.")
        return

    st.info(f"Starting upload for {len(eeg_data)} data points and {len(annotations)} annotations...")

    try:
        # --- 1. Get Firebase clients ---
        db = get_firestore_client()
        bucket = get_storage_bucket()

        # --- 2. Prepare data and filename ---
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eeg_recording_{timestamp}.json"

        upload_data = {
            "eeg_data": eeg_data,
            "annotations": annotations,
            "timestamp": timestamp
        }

        # Convert data to a JSON string
        json_data = json.dumps(upload_data, indent=4)

        # --- 3. Upload to Firebase Storage ---
        with st.spinner(f"Uploading {filename} to Firebase Storage..."):
            blob = bucket.blob(f"recordings/{filename}")
            blob.upload_from_string(json_data, content_type='application/json')
            st.success("EEG data successfully uploaded to Storage.")

        # --- 4. Add metadata to Firestore ---
        with st.spinner("Adding metadata to Firestore..."):
            # Get the public URL of the uploaded file
            public_url = blob.public_url

            # Create a document in the 'recordings' collection
            doc_ref = db.collection('recordings').document(timestamp)
            doc_ref.set({
                'filename': filename,
                'storage_path': f"recordings/{filename}",
                'public_url': public_url,
                'data_points': len(eeg_data),
                'annotation_count': len(annotations),
                'created_at': datetime.datetime.now(datetime.timezone.utc)
            })
            st.success("Recording metadata saved to Firestore.")

        st.balloons()
        st.success(f"Upload complete! View your data at: {public_url}")

    except Exception as e:
        st.error(f"An error occurred during upload: {e}")
        st.error("Please ensure your Firebase credentials and storage bucket are correctly configured.")
