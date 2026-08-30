import streamlit as st

class CameraInput:
    
    @staticmethod
    def camera_input():
        enable = st.checkbox("Activar camara")
        picture = st.camera_input("Toma una foto", disabled=not enable)

        if picture:
            st.image(picture, caption="Capturar imagen")
        
        return picture