import streamlit as st
from PIL import Image
import numpy as np
from super_image import EdsrModel, ImageLoader

# Page Config
st.set_page_config(page_title="AI Image Enhancer", layout="wide")
st.title("🖼️ Image Quality & Clarity Enhancer")
st.write("Fix pixelation and low-resolution issues using Deep Learning.")

# Sidebar Settings
upscale_factor = st.sidebar.selectbox("Upscale Factor", [2, 3, 4], index=1)
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model(scale):
    # This downloads the pre-trained weights from Hugging Face automatically
    model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=scale)
    return model

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original (Low Res)")
        st.image(input_image, use_container_width=True)

    if st.button("🚀 Enhance Clarity"):
        with st.spinner(f"Upscaling {upscale_factor}x..."):
            # Load Model
            model = load_model(upscale_factor)
            
            # Process Image
            inputs = ImageLoader.load_image(input_image)
            preds = model(inputs)
            
            # Display Result
            with col2:
                st.subheader(f"Enhanced ({upscale_factor}x)")
                # super-image handles the conversion back to PIL automatically
                # but we use save/show logic for Streamlit
                st.image(preds, use_container_width=True)
                
                # Simple path to allow download
                st.success("Enhancement Complete!")
