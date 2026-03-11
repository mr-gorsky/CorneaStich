import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

st.image("Phantasmed-logo.png", width=350)

st.title("Peripheral Corneal Topography Mosaic")

st.warning(
"""
DISCLAIMER

This software is an experimental research prototype developed for the study:

'Peripheral Corneal Topography Reconstruction using Multi-Fixation Placido Imaging'

The output is an approximate visualization and MUST NOT be used for
clinical diagnosis or treatment decisions.
"""
)

st.header("Upload topography images")

c1,c2,c3,c4,c5 = st.columns(5)

central_file = c1.file_uploader("central", type=["jpg","png"])
up_file = c2.file_uploader("up", type=["jpg","png"])
down_file = c3.file_uploader("down", type=["jpg","png"])
left_file = c4.file_uploader("left", type=["jpg","png"])
right_file = c5.file_uploader("right", type=["jpg","png"])


def read_img(file):
    img = Image.open(file)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def crop_map(img):

    h,w = img.shape[:2]

    x1 = int(w*0.20)
    x2 = int(w*0.80)

    y1 = int(h*0.05)
    y2 = int(h*0.95)

    return img[y1:y2, x1:x2]


def align_image(base, img):

    orb = cv2.ORB_create(5000)

    g1 = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(g1,None)
    kp2, des2 = orb.detectAndCompute(g2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)

    matches = sorted(matches,key=lambda x:x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:200]]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:200]]).reshape(-1,1,2)

    H,_ = cv2.findHomography(dst_pts,src_pts,cv2.RANSAC,5.0)

    h,w = base.shape[:2]

    aligned = cv2.warpPerspective(img,H,(w,h))

    return aligned


def blend(base,img):

    mask = (img>0)

    base[mask] = img[mask]

    return base


if st.button("Generate Mosaic"):

    if None in [central_file,up_file,down_file,left_file,right_file]:

        st.error("Upload all 5 images")

    else:

        central = crop_map(read_img(central_file))
        up = crop_map(read_img(up_file))
        down = crop_map(read_img(down_file))
        left = crop_map(read_img(left_file))
        right = crop_map(read_img(right_file))

        mosaic = central.copy()

        for img in [up,down,left,right]:

            aligned = align_image(mosaic,img)

            mosaic = blend(mosaic,aligned)

        st.image(cv2.cvtColor(mosaic,cv2.COLOR_BGR2RGB))

        result = Image.fromarray(cv2.cvtColor(mosaic,cv2.COLOR_BGR2RGB))

        st.download_button(
            "Download Mosaic",
            data=result.tobytes(),
            file_name="corneal_mosaic.png",
            mime="image/png"
        )
