#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import streamlit as st
from PIL import Image

def colorizer(img):
    # Handle different input image types
    if len(img.shape) == 2:  # grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # else assume it's already RGB

    # Load the colorization model
    prototxt = r"C:\Users\KAVIN KUMAR S\img_clr\Colorizer\models\models_colorization_deploy_v2.prototxt"
    model = r"C:\Users\KAVIN KUMAR S\img_clr\Colorizer\models\colorization_release_v2.caffemodel"
    points = r"C:\Users\KAVIN KUMAR S\img_clr\Colorizer\models\pts_in_hull.npy"

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # Add cluster centers to the network
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Convert image to LAB
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    # Resize to 224x224 and extract L channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Pass through network to predict AB channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize AB channels to original image size
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    # Combine original L channel with predicted AB channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert back to RGB and clip values
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

##########################################################################################################

st.write("""
          # Colorize your Black and white image
          """)

st.write("This app turns your B&W images into color images.")

file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)

    st.text("Your original image")
    st.image(image, use_column_width=True)

    st.text("Your colorized image")
    color = colorizer(img)

    st.image(color, use_column_width=True)

    print("done!")
