import streamlit as st 
from PIL import Image
import tensorflow as tf 
import numpy as np 



# load the model 
model = tf.keras.models.load_model('./Model/Dog_Cat_classifier.h5')

# Define the class labels 
class_labels = ['Cat', 'Dog']

# welcome message 
st.title("Image Classifier")
st.subheader(":wave: Welcome to the Image Classification Web App!")

upload_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if upload_image is not None:
    try:
        # preprocess the image 
        image = Image.open(upload_image)
        image = image.resize((256,256))
        image_array = np.array(image) / 255
        image_array = np.expand_dims(image_array, axis=0)

        # make predictions 
        predictions = model.predict(image_array)
        predicted_class = class_labels[int(np.round(predictions[0]))]

        # display the image 
        st.image(image, use_column_width=True, caption=f"Class: {predicted_class}")
        st.write(f""" 
            Predicted Class : {predicted_class} \n
            Probability: {predictions[0][0]:.2f}
        """)
    except Exception as e:
        st.error(f"Error processing the image : {str(e)}")
