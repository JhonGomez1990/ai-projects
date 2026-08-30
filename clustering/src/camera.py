import streamlit as st

class CameraInput:
    
    @staticmethod
    def camera_input():
        enable = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable)

        if picture:
            st.image(picture, caption="Capture image")
        
        return picture