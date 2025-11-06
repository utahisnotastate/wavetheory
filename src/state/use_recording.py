
import streamlit as st
import asyncio
import threading
import queue
import json

def use_recording():
    # Initialize all session state variables at the beginning
    st.session_state.setdefault('recording', False)
    st.session_state.setdefault('eeg_data', [])
    st.session_state.setdefault('annotations', [])
    st.session_state.setdefault('data_queue', queue.Queue())
    st.session_state.setdefault('streaming_thread', None)
    st.session_state.setdefault('is_streaming', False)

    def _stream_worker(cortex_client, token, session_id, data_queue):
        """Worker function to run in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def subscribe_and_listen():
            await cortex_client.subscribe(token, session_id, ["eeg"])
            while st.session_state.get('is_streaming', False):
                try:
                    # Set a timeout to prevent indefinite blocking
                    message = await asyncio.wait_for(cortex_client.websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    if "eeg" in data:
                        data_queue.put(data["eeg"])
                except asyncio.TimeoutError:
                    # Continue loop to check the streaming flag
                    continue
                except Exception as e:
                    st.error(f"Error in stream listener: {e}")
                    break

        loop.run_until_complete(subscribe_and_listen())

    def start_streaming(cortex_client, token, headset_id):
        if not st.session_state.is_streaming:
            try:
                session_id = asyncio.run(cortex_client.create_session(token, headset_id))
                if not session_id:
                    st.error("Failed to create a session.")
                    return

                thread = threading.Thread(
                    target=_stream_worker,
                    args=(cortex_client, token, session_id, st.session_state.data_queue),
                    daemon=True
                )
                st.session_state.is_streaming = True
                st.session_state.streaming_thread = thread
                thread.start()
                st.success("Data stream started.")
            except Exception as e:
                st.error(f"Failed to start stream: {e}")


    def stop_streaming():
        if st.session_state.is_streaming:
            st.session_state.is_streaming = False
            if st.session_state.streaming_thread:
                st.session_state.streaming_thread.join(timeout=5)
            st.session_state.streaming_thread = None
            st.info("Data stream stopped.")

    def start_recording():
        st.session_state.recording = True
        st.session_state.eeg_data = []
        st.session_state.annotations = []

    def stop_recording():
        st.session_state.recording = False

    def process_eeg_data():
        """Process data from the queue and add it to the main data list."""
        while not st.session_state.data_queue.empty():
            data = st.session_state.data_queue.get_nowait()
            if st.session_state.recording:
                st.session_state.eeg_data.append(data)

    def add_annotation(annotation):
        if st.session_state.recording:
            st.session_state.annotations.append(annotation)

    return {
        "recording": st.session_state.recording,
        "is_streaming": st.session_state.is_streaming,
        "eeg_data": st.session_state.eeg_data,
        "annotations": st.session_state.annotations,
        "start_recording": start_recording,
        "stop_recording": stop_recording,
        "add_annotation": add_annotation,
        "start_streaming": start_streaming,
        "stop_streaming": stop_streaming,
        "process_eeg_data": process_eeg_data
    }
