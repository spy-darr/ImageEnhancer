import streamlit as st
import cv2
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch

# --- Setup Models ---
def load_models():
    # Use RRDBNet for Real-ESRGAN
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Pre-trained Real-ESRGAN for general enhancement
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False
    )
    
    # Face restoration model
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler
    )
    return upsampler, face_enhancer

st.set_page_config(page_title="AI Image Restorer", layout="wide")
st.title("🖼️ Image Clarity & Restoration Tool")
st.write("Upload an old or pixelated image to enhance it using AI.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
enhance_faces = st.sidebar.checkbox("Enhance Faces (Best for old photos)", value=True)

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img, use_column_width=True)

    if st.button("✨ Restore Image"):
        with st.spinner("Processing... this may take a moment."):
            upsampler, face_enhancer = load_models()
            
            if enhance_faces:
                # Restores faces + upscales background
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                # General upscaling (best for pixelation/objects)
                output, _ = upsampler.enhance(img, outscale=4)

            with col2:
                st.subheader("Enhanced Image")
                st.image(output, use_column_width=True)
                
                # Download button
                result_img = Image.fromarray(output)
                st.download_button("Download High-Res Image", data=output.tobytes(), file_name="enhanced.png", mime="image/png")
