
import streamlit as st
from state.use_recording import use_recording

def annotator():
    recording_state = use_recording()

    st.header("Annotations")

    annotation_text = st.text_input("Enter annotation:")

    if st.button("Add Annotation", disabled=not recording_state["recording"]):
        if annotation_text:
            recording_state["add_annotation"](annotation_text)
            st.success(f"Added annotation: '{annotation_text}'")
            st.session_state.annotation_text = "" # Clear input
        else:
            st.warning("Please enter an annotation.")

    st.write("Annotations:", recording_state["annotations"])
