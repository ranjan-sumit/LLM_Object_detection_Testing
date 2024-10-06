import os
import base64
import streamlit as st
from mistralai import Mistral
from dotenv import load_dotenv
import os
import base64
import streamlit as st
from mistralai import Mistral
import requests as r
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
st.sidebar.title('Styra Model Testing App')
st.sidebar.write('This model is created for testing the different object detection model for meter image testing')
# Load environment variables
load_dotenv()
api_key = "SzjdK8qBgUAHpCkmsjb5ku9S9WJjadxX"

# Initialize Mistral client
model = "pixtral-12b-2409"
client = Mistral(api_key=api_key)

# Function to encode image to base64
def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Streamlit app
st.title("Styra Image Analysis")
uploaded_file = st.file_uploader("Upload an image of a digital electricity meter", type=["jpeg", "jpg"])

if uploaded_file is not None:
    # Read the uploaded image and encode it
    base_64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")

    prompt = """
    You’re an advanced image analysis assistant with a strong background in interpreting images of digital devices like electric meters. You excel at reading labels from various instruments and extracting relevant data in a clear and structured format. Your expertise allows you to accurately identify and extract values from complex images, ensuring precision and clarity.

    Your task is to analyze an image of a digital electricity meter and extract the required details. 
    Here are the details I’d like you to keep in mind:
    - The image will contain a digital electricity meter with various information
    - You need to capture the below details and give the output in a single JSON format. 
        Serial Number: Identify the unique serial number printed or engraved on the meter. It is alphanumeric starting with alphabets followed by numbers
        Meter Reading: Capture the current meter reading which is present in the digital display. Do not look for it in any other part of the image. Also pick the unit if it is available
        Phase Information: Determine the phase (e.g., single-phase or three-phase) based on any labels, symbols, or text on the meter.
        Meter Type: Recognize the type of meter (e.g., Smart  or Normal) based on visual cues.
        Billing Type: Recognize the Billing type (e.g., PrePaid  or PostPaid) based on visual cues or presence of keywords.
        Net Meter Type Validation: Recognize the type (e.g., Bidirectional  or NA) based on visual cues or presence of keywords.

    - Focus on accuracy and ensure all relevant readings are captured.
    - Do not provide any additional explanation 
    """

   # Call the Pixtral model
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base_64_image}"}
                ]
            }
        ]
    )

    # Function to encode image to base64 for Pixtral
    def encode_image_base64(image):
        return base64.b64encode(image.read()).decode("utf-8")
    
    # Function to call YOLO OCR model
    def get_yolo_ocr_prediction(image):
        url = "https://7hezksr22pptk3-8000.proxy.runpod.net/predict"
        files = {'file': image}
        response = r.post(url, files=files)
        return response.json()
    
    # Function to call GOT OCR model
    def get_got_ocr_prediction(image):
        url = "https://7hezksr22pptk3-8004.proxy.runpod.net/predict"
        files = {'image_file': image}
        response = r.post(url, files=files)
        return response.json()

    
    # Debugging: Print the raw response
    st.write("Raw Pixtral Response:")
    st.write(chat_response.choices[0].message.content)

        # Reset the file pointer to the start for the next upload
    uploaded_file.seek(0)

    # YOLO OCR prediction
    yolo_ocr_prediction = get_yolo_ocr_prediction(uploaded_file)
    st.write("YOLO OCR Prediction:")
    st.json(yolo_ocr_prediction)

    # Reset the file pointer to the start for the next upload
    uploaded_file.seek(0)

    # GOT OCR prediction
    got_ocr_prediction = get_got_ocr_prediction(uploaded_file)
    st.write("GOT OCR Prediction:")
    st.json(got_ocr_prediction)

    if uploaded_file is not None:
         # Load the image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
