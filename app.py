import streamlit as st
from PIL import Image
import torch
from super_image import EdsrModel, ImageLoader

# Page Config
st.set_page_config(page_title="AI Image Enhancer", layout="wide")
st.title("🖼️ Image Quality & Clarity Enhancer")
st.write("Fix pixelation and low-resolution issues using AI.")

# Sidebar Settings
upscale_factor = st.sidebar.selectbox("Upscale Factor", [2, 3, 4], index=1)
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model(scale):
    # Using EDSR model which is great for pixelation
    model = EdsrModel.from_pretrained('eugenesiow/edsr', scale=scale)
    return model

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(input_image, width='stretch')

    if st.button("🚀 Enhance Clarity"):
        with st.spinner(f"Upscaling {upscale_factor}x..."):
            try:
                # Load Model
                model = load_model(upscale_factor)
                
                # Process Image
                inputs = ImageLoader.load_image(input_image)
                preds = model(inputs)
                
                # Display Result
                with col2:
                    st.subheader(f"Enhanced ({upscale_factor}x)")
                    st.image(preds, width='stretch')
                    st.success("Enhancement Complete!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
