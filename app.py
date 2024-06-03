import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Image Description Chatbot by Satyam")

# Upload image file
uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Generate image caption
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = blip_model.generate(**inputs)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    
    st.session_state.history.append(("Image Caption", caption))
    st.write(f"Image caption: {caption}")

# Text input for general chatbot
user_input = st.text_input("You: ", "")
if st.button("Send"):
    if user_input:
        st.write(f"You: {user_input}")

        # Handle image-related questions
        if "image" in user_input.lower() and any("Image Caption" in item[0] for item in st.session_state.history):
            image_caption = next(item[1] for item in st.session_state.history if item[0] == "Image Caption")
            st.write(f"Bot: The image appears to be: {image_caption}. Let's discuss it further!")
