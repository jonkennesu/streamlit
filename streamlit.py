import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

st.write("""
# Weather Classification System""")
file = st.file_uploader("Choose weather photo from computer", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
  size = (256,256)
  image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  
  img = np.asarray(image)
  img_reshape = img[np.newaxis,...]
  prediction = model.predict(img_reshape)
  return prediction

if file is None:
  st.text("Please upload an image file")
else:
  image = Image.open(file)
  st.image(image, use_container_width=True)  
  prediction = import_and_predict(image, model)
  class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
  string = "The output is: " + class_names[np.argmax(prediction)]
  st.success(string)