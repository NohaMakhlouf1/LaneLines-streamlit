import streamlit as st
from model import preprocess
import numpy as np
import cv2

from PIL import Image, ImageEnhance
import streamlit as st

st.header(" Road Lines Detector App ")
image_path = st.text_input(" Enter Image Path : ")

submit = st.button("Submit")
if submit:
    result = preprocess(image_path)
    img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    st.image(img)


