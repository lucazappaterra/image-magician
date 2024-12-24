import streamlit as st
from dotenv import load_dotenv
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import cv2
import numpy as np
from components.create_mask import create_mask
from components.prompts import positive_prompts, negative_prompt_inpainting

load_dotenv()

st.set_page_config(page_title="Image Magician", page_icon="🪄", layout="centered", initial_sidebar_state="auto")
st.title("🪄Image Magician🪄")

if 'user_photo' not in st.session_state:
    st.session_state.user_photo = None
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False
if 'show_uploader' not in st.session_state:
    st.session_state.show_uploader = False
if 'inpainted_image' not in st.session_state:
    st.session_state.inpainted_image = None
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'device' not in st.session_state:
    st.session_state.device = None

# Load the Stable Diffusion model
if 'pipe' not in st.session_state:
    if not torch.cuda.is_available():
        st.session_state.device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        st.session_state.device = "cuda"
    st.session_state.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "lykon/absolute-reality-1.6525-inpainting", #stabilityai/stable-diffusion-2-inpainting
    #  'Lykon/dreamshaper-8-inpainting',
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to(st.session_state.device)
    # st.session_state.pipe.enable_model_cpu_offload()

with st.expander("ℹ️ About"):
    st.write("This is a simple image processing app that allows you to create a fancy version of yourself with AI image generation tools.")

col1, col2 = st.columns((1,3), gap="small")

with col1:
    if st.button("Take a photo"):
        st.session_state.show_camera = True
        st.session_state.show_uploader = False
        
    if st.button("Upload a photo"):
        st.session_state.show_uploader = True
        st.session_state.show_camera = False
    
    if st.button("Clear photo"):
        st.session_state.user_photo = None
        st.session_state.inpainted_image = None

with col2:
    if st.session_state.show_camera:
        photo = st.camera_input("Let's take a photo!")
        if photo is not None:
            st.session_state.user_photo = photo
            st.session_state.photo_is_from_camera = True
            st.session_state.show_camera = False
            
    if st.session_state.show_uploader:
        uploaded_file = st.file_uploader("Upload your photo")
        if uploaded_file is not None:
            st.session_state.user_photo = uploaded_file
            st.session_state.photo_is_from_camera = False
            st.session_state.show_uploader = False

selected_prompt_key = st.selectbox("Choose a prompt", list(positive_prompts.keys()))
prompt_inpainting = positive_prompts[selected_prompt_key]
seed = st.slider("Seed", -1, 1000, -1, 1)
st.session_state.generator = torch.Generator(device=st.session_state.device).manual_seed(seed)


st.write("👇🏻 Scroll down to see the magic happen!") 

with st.expander("🪄 Magic", expanded=True):
    if st.session_state.user_photo:
        # st.image(st.session_state.user_photo, caption="Your Image", use_container_width=True)
        uploaded_image = st.session_state.user_photo
        image_bytes = uploaded_image.getvalue()
        img_cv = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) # convert to RGB

        mask = create_mask(img_cv)

        init_image = Image.fromarray(img_cv).convert("RGB")
        original_width, original_height = init_image.size
        print(f"Original dimensions: {original_width}x{original_height}")
        
        max_size = (768, 768)
        init_image.thumbnail(max_size, Image.Resampling.LANCZOS)
        # init_image.show()
        down_width, down_height = init_image.size

        output_width = down_width - (down_width % 8)
        output_height = down_height - (down_height % 8)
        print(f"Downscaled to: {output_width}x{output_height}")

        fill = Image.new("RGB", init_image.size, "black")

    if st.session_state.inpainted_image is None:
        
        # Generate the transformed image
        if st.button("Generate Image"):
            with st.spinner("Generating image..."):
                output_filename = "inpainted_image.png"
                inpainted_image = st.session_state.pipe(
                    prompt=prompt_inpainting, 
                    negative_prompt=negative_prompt_inpainting,
                    image=init_image, 
                    mask_image=mask, 
                    init_image = fill,
                    height=output_height,#960
                    width=output_width,#560
                    generator=st.session_state.generator
                    # strength=0.99, 
                    # num_inference_steps=4,
                    # guidance_scale=0.5
                ).images[0]
                st.session_state.inpainted_image = inpainted_image
                inpainted_image.save(output_filename)
                print(f"Image saved as: {output_filename}")
        
        # Display the transformed image
    if st.session_state.inpainted_image is not None:
        st.image(st.session_state.inpainted_image, caption="Transformed Image", use_container_width=True)