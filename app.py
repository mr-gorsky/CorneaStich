import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Corneal Topography Mosaic", layout="wide")

st.image("Phantasmed-logo.png", width=300)

st.title("Corneal Topography Peripheral Mosaic")

st.markdown(
"""
### Research Tool

This application generates an **extended corneal topography mosaic**
from multiple fixation Placido images.

Images required:

- central
- up
- down
- left
- right

The algorithm crops the corneal map and generates an approximate mosaic
of peripheral corneal curvature.

"""
)

st.warning(
"""
**DISCLAIMER**

This software is an experimental research tool developed for the study:

*Peripheral Corneal Topography Reconstruction using Multi-Fixation Placido Imaging*

It is **NOT intended for clinical decision making** and must not be used
for diagnosis or treatment planning.
"""
)

st.header("Upload Images")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    central_file = st.file_uploader("Central", type=["jpg","png"])

with col2:
    up_file = st.file_uploader("Up (gaze up)", type=["jpg","png"])

with col3:
    down_file = st.file_uploader("Down (gaze down)", type=["jpg","png"])

with col4:
    left_file = st.file_uploader("Left (gaze left)", type=["jpg","png"])

with col5:
    right_file = st.file_uploader("Right (gaze right)", type=["jpg","png"])


def load_image(file):
    image = Image.open(file)
    return np.array(image)


def crop_corneal_map(img):

    h, w = img.shape[:2]

    # central crop region (removes legends)
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    y1 = int(h * 0.1)
    y2 = int(h * 0.9)

    crop = img[y1:y2, x1:x2]

    return crop


def build_mosaic(images):

    base = images["central"]

    h, w = base.shape[:2]

    canvas = np.zeros((h*3, w*3, 3), dtype=np.uint8)

    cx = w
    cy = h

    canvas[cy:cy+h, cx:cx+w] = base

    canvas[0:h, cx:cx+w] = images["down"]   # superior cornea
    canvas[h*2:h*3, cx:cx+w] = images["up"] # inferior cornea

    canvas[cy:cy+h, 0:w] = images["right"]  # nasal
    canvas[cy:cy+h, w*2:w*3] = images["left"] # temporal

    return canvas


if st.button("Generate Mosaic"):

    if None in [central_file, up_file, down_file, left_file, right_file]:

        st.error("Upload all 5 images.")

    else:

        central = crop_corneal_map(load_image(central_file))
        up = crop_corneal_map(load_image(up_file))
        down = crop_corneal_map(load_image(down_file))
        left = crop_corneal_map(load_image(left_file))
        right = crop_corneal_map(load_image(right_file))

        images = {
            "central": central,
            "up": up,
            "down": down,
            "left": left,
            "right": right
        }

        mosaic = build_mosaic(images)

        st.image(mosaic, caption="Generated Corneal Mosaic")

        result = Image.fromarray(mosaic)

        st.download_button(
            label="Download Mosaic",
            data=result.tobytes(),
            file_name="corneal_mosaic.png",
            mime="image/png"
        )
