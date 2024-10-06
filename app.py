# import streamlit as st
# import gdown
# import os
# import json
# from ultralytics import YOLO
# from PIL import Image
# import tempfile

# # Function to download the YOLO model from Google Drive
# def download_model():
#     url = 'https://drive.google.com/uc?id=1F8J0iJCu6Gq_FahyEHDTbBybo6eUCk1Y'
#     output = '700model.pt'
#     gdown.download(url, output, quiet=False)

# # Download the YOLO model
# download_model()

# # Load the model
# model = YOLO("700model.pt")

# # Class mapping
# class_mapping = {
#     0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5',
#     7: '6', 8: '7', 9: '8', 10: '9', 11: 'A', 12: 'Date', 13: 'F',
#     14: 'MD', 15: 'P', 16: 'U', 17: 'Y', 18: 'b', 19: 'box', 20: 'h',
#     21: 'kW', 22: 'kv', 23: '.', 24: 'v'
# }

# # Streamlit app layout
# st.title("YOLO Model Image Prediction")
# st.write("Upload an image for prediction.")

# # Image uploader
# uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load the image
#     img = Image.open(uploaded_file).convert("RGB")

#     # Display the image
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     # Save the image to a temporary file
#     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_image_file:
#         img.save(tmp_image_file.name)

#         # Run inference
#         results = model(tmp_image_file.name)

#     predictions = []
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             bbox = box.xyxy[0].tolist()
#             confidence = box.conf[0].item()
#             class_id = box.cls[0].item()

#             predictions.append({
#                 "class_id": int(class_id),
#                 "bbox": bbox,
#                 "confidence": confidence
#             })

#     predictions = sorted(predictions, key=lambda x: x['bbox'][0])
#     mapped_values = [class_mapping[pred['class_id']] for pred in predictions]

#     reading = ""
#     unit = "kwh"  # Default unit

#     for value in mapped_values:
#         if value in ["kv", "kW", "v", "A"]:
#             unit = value
#         else:
#             reading += value

#     # Display results
#     st.write("Prediction Results:")
#     st.json({
#         "file": uploaded_file.name,
#         "reading": reading,
#         "unit": unit
#     })

#     # Clean up temporary files
#     os.remove(tmp_image_file.name)  # Delete the temporary file

import os
import base64
import streamlit as st
from mistralai import Mistral
from dotenv import load_dotenv

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
st.title("Pixtral Image Analysis")
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
    
    # Debugging: Print the raw response
    st.write("Raw Pixtral Response:")
    st.write(chat_response.choices[0].message.content)
    
    # # Attempt to parse the JSON
    # try:
    #     json_output = chat_response.choices[0].message.content
    #     # Display as JSON if valid
    #     st.json(json_output)
    # except Exception as e:
    #     st.error(f"Error parsing JSON: {e}")


    # # Display the response
    # st.write("Pixtral Prediction Results:")
    # st.json(chat_response.choices[0].message.content)

# import streamlit as st
# import gdown
# import os
# import json
# import base64
# from ultralytics import YOLO
# from PIL import Image
# import tempfile
# from mistralai import Mistral
# from dotenv import load_dotenv

# st.sidebar.title('Model Testing App')
# st.sidebar.write('This model is created for testing the different object detection model for meter image testing')

# # Load environment variables
# load_dotenv()
# api_key = "SzjdK8qBgUAHpCkmsjb5ku9S9WJjadxX"
# model_pixtral = "pixtral-12b-2409"
# client = Mistral(api_key=api_key)

# # Function to download the YOLO model from Google Drive
# def download_model():
#     url = 'https://drive.google.com/uc?id=1F8J0iJCu6Gq_FahyEHDTbBybo6eUCk1Y'
#     output = '700model.pt'
#     gdown.download(url, output, quiet=False)

# # # Download and load the YOLO model
# # download_model()
# # model_yolo = YOLO("700model.pt")

# # Class mapping for YOLO
# class_mapping = {
#     0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5',
#     7: '6', 8: '7', 9: '8', 10: '9', 11: 'A', 12: 'Date', 13: 'F',
#     14: 'MD', 15: 'P', 16: 'U', 17: 'Y', 18: 'b', 19: 'box', 20: 'h',
#     21: 'kW', 22: 'kv', 23: '.', 24: 'v'
# }

# # Streamlit app layout
# st.title("Image Prediction using YOLO and Pixtral")

# # Dropdown to select model type
# model_type = st.selectbox("Select Model Type", ["YOLO", "Pixtral"])

# if model_type == "YOLO":
#     st.write("Upload an image for prediction using YOLO.")
#     uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
#     # Download and load the YOLO model
#     download_model()
#     model_yolo = YOLO("700model.pt")

#     if uploaded_file is not None:
#         # Load the image
#         img = Image.open(uploaded_file).convert("RGB")
#         #st.image(img, caption="Uploaded Image", use_column_width=True)

#         # Save the image to a temporary file
#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_image_file:
#             img.save(tmp_image_file.name)

#             # Run inference
#             results = model_yolo(tmp_image_file.name)

#         predictions = []
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 bbox = box.xyxy[0].tolist()
#                 confidence = box.conf[0].item()
#                 class_id = box.cls[0].item()

#                 predictions.append({
#                     "class_id": int(class_id),
#                     "bbox": bbox,
#                     "confidence": confidence
#                 })

#         predictions = sorted(predictions, key=lambda x: x['bbox'][0])
#         mapped_values = [class_mapping[pred['class_id']] for pred in predictions]

#         reading = ""
#         unit = "kwh"  # Default unit

#         for value in mapped_values:
#             if value in ["kv", "kW", "v", "A"]:
#                 unit = value
#             else:
#                 reading += value

#         # Display results
#         st.write("Prediction Results:")
#         st.json({
#             "file": uploaded_file.name,
#             "reading": reading,
#             "unit": unit
#         })
#         st.image(img, caption="Uploaded Image", use_column_width=True)
#         # Clean up temporary files
#         os.remove(tmp_image_file.name)  # Delete the temporary file

# elif model_type == "Pixtral":
#     # Load environment variables
#         load_dotenv()
#         api_key = "SzjdK8qBgUAHpCkmsjb5ku9S9WJjadxX"
        
#         # Initialize Mistral client
#         model = "pixtral-12b-2409"
#         client = Mistral(api_key=api_key)
        
#         # Function to encode image to base64
#         def encode_image_base64(image_path):
#             with open(image_path, "rb") as image_file:
#                 return base64.b64encode(image_file.read()).decode("utf-8")
        
#         # Streamlit app
#         st.title("Pixtral Image Analysis")
#         uploaded_file = st.file_uploader("Upload an image of a digital electricity meter", type=["jpeg", "jpg"])


            
#         if uploaded_file is not None:
#             # Read the uploaded image and encode it
#             base_64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")
        
#             prompt = """
#             You’re an advanced image analysis assistant with a strong background in interpreting images of digital devices like electric meters. You excel at reading labels from various instruments and extracting relevant data in a clear and structured format. Your expertise allows you to accurately identify and extract values from complex images, ensuring precision and clarity.
        
#             Your task is to analyze an image of a digital electricity meter and extract the required details. 
#             Here are the details I’d like you to keep in mind:
#             - The image will contain a digital electricity meter with various information
#             - You need to capture the below details and give the output in a single JSON format. 
#                 Serial Number: Identify the unique serial number printed or engraved on the meter. It is alphanumeric starting with alphabets followed by numbers
#                 Meter Reading: Capture the current meter reading which is present in the digital display. Do not look for it in any other part of the image. Also pick the unit if it is available
#                 Phase Information: Determine the phase (e.g., single-phase or three-phase) based on any labels, symbols, or text on the meter.
#                 Meter Type: Recognize the type of meter (e.g., Smart  or Normal) based on visual cues.
#                 Billing Type: Recognize the Billing type (e.g., PrePaid  or PostPaid) based on visual cues or presence of keywords.
#                 Net Meter Type Validation: Recognize the type (e.g., Bidirectional  or NA) based on visual cues or presence of keywords.
        
#             - Focus on accuracy and ensure all relevant readings are captured.
#             - Do not provide any additional explanation 
#             """
        
#            # Call the Pixtral model
#             chat_response = client.chat.complete(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": prompt},
#                             {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base_64_image}"}
#                         ]
#                     }
#                 ]
#             )
            
#             # Debugging: Print the raw response
#             st.write("Raw Pixtral Response:")
#             st.write(chat_response.choices[0].message.content)


#             if uploaded_file is not None:
#             # Load the image
#                 img = Image.open(uploaded_file).convert("RGB")
#                 st.image(img, caption="Uploaded Image", use_column_width=True)
                
