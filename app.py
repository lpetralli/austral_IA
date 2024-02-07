from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st 
import openai
from openai import OpenAI



def classify_waste(img):

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

    return class_name, confidence_score


def generate_carbon_footprint_info(label):

    
    

    client = OpenAI(api_key="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")

    

    response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt= f"You are a helpful assistant with experience in Carbon Emissions. What is the approximate Carbon emission or carbon footprint generated from {label}. I just need an approximate number to create awa reness. Elaborate in 100 words.",
    temperature=1,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response.choices[0].text





# Streamlit App

st.set_page_config(layout='wide')

st.title("Waste Classifier Sustainability App")

input_img = st.file_uploader("Enter your image", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Classify"):
        
        col1, col2, col3 = st.columns([1,1,1])

        with col1:
            st.info("Your uploaded Image")
            st.image(input_img, use_column_width=True)

        with col2:
            st.info("Your Result")
            image_file = Image.open(input_img)
            label, confidence_score = classify_waste(image_file)
            print(label)
            # col4, col5 = st.columns([1,1])
            if label == "4 cardboard\n":
                st.success("The image is classified as CARDBOARD.")                
                # with col4:
                #     st.image("sdg goals/12.png", use_column_width=True)
                #     st.image("sdg goals/13.png", use_column_width=True)
                # with col5:
                #     st.image("sdg goals/14.png", use_column_width=True)
                #     st.image("sdg goals/15.png", use_column_width=True) 
            elif label == "0 plastic\n":
                st.success("The image is classified as PLASTIC.")
                # with col4:
                #     st.image("sdg goals/6.jpg", use_column_width=True)
                #     st.image("sdg goals/12.png", use_column_width=True)
                # with col5:
                #     st.image("sdg goals/14.png", use_column_width=True)
                #     st.image("sdg goals/15.png", use_column_width=True) 
            elif label == "3 glass\n":
                st.success("The image is classified as GLASS.")
                # with col4:
                #     st.image("sdg goals/12.png", use_column_width=True)
                # with col5:
                #     st.image("sdg goals/14.png", use_column_width=True)
            elif label == "2 metal\n":
                st.success("The image is classified as METAL.")
                # with col4:
                #     st.image("sdg goals/3.png", use_column_width=True)
                #     st.image("sdg goals/6.jpg", use_column_width=True)
                # with col5:
                #     st.image("sdg goals/12.png", use_column_width=True)
                #     st.image("sdg goals/14.png", use_column_width=True) 
            else:
                st.error("The image is not classified as any relevant class.")

        with col3:
            result = generate_carbon_footprint_info(label)
            st.success(result)




