
import streamlit as st
import asyncio
from src.hooks.use_cortex import CortexClient

def device_card():
    st.header("Device Connection")

    # Initialize session state for connection management
    if 'cortex_client' not in st.session_state:
        st.session_state.cortex_client = None
    if 'cortex_token' not in st.session_state:
        st.session_state.cortex_token = None
    if 'is_connected' not in st.session_state:
        st.session_state.is_connected = False

    if not st.session_state.is_connected:
        st.subheader("Cortex API Credentials")
        client_id = st.text_input("Client ID", "")
        client_secret = st.text_input("Client Secret", "", type="password")

        headset = st.selectbox("Select Headset", ["Emotiv Insight", "Emotiv EPOC+", "Unicorn"])

        if st.button("Connect"):
            if not client_id or not client_secret:
                st.error("Please enter both Client ID and Client Secret.")
            else:
                with st.spinner("Connecting to Cortex service..."):
                    try:
                        client = CortexClient(client_id, client_secret)
                        # Use asyncio.run() to execute async functions
                        asyncio.run(client.connect())
                        token = asyncio.run(client.authorize())

                        if token:
                            st.session_state.cortex_client = client
                            st.session_state.cortex_token = token
                            st.session_state.is_connected = True
                            st.success("Successfully connected and authorized with Cortex!")
                        else:
                            st.error("Authorization failed. Please check your credentials.")
                            # Clean up client if connection was made but auth failed
                            if client.websocket:
                                asyncio.run(client.close())
                    except Exception as e:
                        st.error(f"Failed to connect: {e}")
    else:
        st.success("Device is connected.")
        st.write(f"Cortex Token: `{st.session_state.cortex_token[:5]}...`")

        if st.button("Disconnect"):
            with st.spinner("Disconnecting..."):
                try:
                    if st.session_state.cortex_client:
                        asyncio.run(st.session_state.cortex_client.close())
                    # Reset session state
                    st.session_state.cortex_client = None
                    st.session_state.cortex_token = None
                    st.session_state.is_connected = False
                    st.info("Disconnected from Cortex.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"An error occurred during disconnection: {e}")
