import streamlit as st
from src.camera import CameraInput

def main():
    st.set_page_config(
        page_title="Image Clustering",
        page_icon="📷",
        layout="centered"
    )

    st.title("Image Clustering")
    st.write("Capture an image using your device camera.")

    picture = CameraInput.camera_input()
    
    if picture is not None:
        st.success("Image captured successfully")


if __name__ == "__main__":
    main()