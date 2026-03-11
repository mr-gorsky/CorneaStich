import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="TopoStitcher - Corneal Topography Mosaicing",
    page_icon="🩺",
    layout="wide"
)

# Učitaj logo
logo_path = Path(__file__).parent / "Phantasmed-logo.png"
if logo_path.exists():
    logo = Image.open(logo_path)
else:
    logo = None

# Header
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if logo:
        st.image(logo, width=150)
    else:
        st.markdown("### 🩺")
with col_title:
    st.title("TopoStitcher")
    st.markdown("##### Corneal Topography Mosaicing for Peripheral Assessment")

st.markdown("---")

def load_image(uploaded_file):
    """Učitaj sliku iz uploadanog fajla"""
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None

def crop_to_circle(image):
    """Izreži samo kružni dio topografske mape"""
    if image is None:
        return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Napravi masku
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Izreži
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Bounding box
        y1 = max(0, center[1] - radius)
        y2 = min(image.shape[0], center[1] + radius)
        x1 = max(0, center[0] - radius)
        x2 = min(image.shape[1], center[0] + radius)
        
        return result[y1:y2, x1:x2]
    
    return None

def create_wide_panorama(images_dict, eye_side):
    """Napravi JEDNU široku sliku od 5 snimki"""
    
    if 'Central fix' not in images_dict:
        return None
    
    central = images_dict['Central fix']
    h, w = central.shape[:2]
    
    # Definiraj redoslijed za široku panoramu (s lijeva na desno)
    if eye_side == "Left eye (OS)":
        # Za lijevo oko: NAZALNO (desni fix) - CENTRAL - TEMPORALNO (lijevi fix)
        order = ['Right fix', 'Central fix', 'Left fix']
    else:
        # Za desno oko: NAZALNO (lijevi fix) - CENTRAL - TEMPORALNO (desni fix)
        order = ['Left fix', 'Central fix', 'Right fix']
    
    # Prvo napravi horizontalnu panoramu (nazalno - centralno - temporalno)
    horizontal_images = []
    for name in order:
        if name in images_dict:
            img = images_dict[name]
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            horizontal_images.append(img)
        else:
            # Ako fali slika, dodaj prazno mjesto
            horizontal_images.append(np.ones((h, w, 3), dtype=np.uint8) * 255)
    
    # Spoji horizontalno
    horizontal = np.hstack(horizontal_images)
    
    # Sada dodaj gornju i donju snimku (ako postoje)
    h_horiz, w_horiz = horizontal.shape[:2]
    
    # Pripremi vertikalne snimke
    top_img = None
    bottom_img = None
    
    if 'Up fix' in images_dict:  # Gleda gore → snima se donji dio
        top_img = images_dict['Up fix']
        if top_img.shape[0] != h or top_img.shape[1] != w:
            top_img = cv2.resize(top_img, (w, h))
    
    if 'Down fix' in images_dict:  # Gleda dolje → snima se gornji dio
        bottom_img = images_dict['Down fix']
        if bottom_img.shape[0] != h or bottom_img.shape[1] != w:
            bottom_img = cv2.resize(bottom_img, (w, h))
    
    # Kreiraj konačnu panoramu
    if top_img is not None and bottom_img is not None:
        # Sve tri: gornja, horizontalna, donja
        top_row = np.ones((h, w_horiz, 3), dtype=np.uint8) * 255
        bottom_row = np.ones((h, w_horiz, 3), dtype=np.uint8) * 255
        
        # Postavi gornju sliku u sredinu
        start_x = (w_horiz - w) // 2
        top_row[:, start_x:start_x+w] = top_img
        bottom_row[:, start_x:start_x+w] = bottom_img
        
        panorama = np.vstack([top_row, horizontal, bottom_row])
        
    elif top_img is not None:
        # Samo gornja + horizontalna
        top_row = np.ones((h, w_horiz, 3), dtype=np.uint8) * 255
        start_x = (w_horiz - w) // 2
        top_row[:, start_x:start_x+w] = top_img
        panorama = np.vstack([top_row, horizontal])
        
    elif bottom_img is not None:
        # Samo horizontalna + donja
        bottom_row = np.ones((h, w_horiz, 3), dtype=np.uint8) * 255
        start_x = (w_horiz - w) // 2
        bottom_row[:, start_x:start_x+w] = bottom_img
        panorama = np.vstack([horizontal, bottom_row])
        
    else:
        # Samo horizontalna
        panorama = horizontal
    
    return panorama

def add_scale_bar(image, hvid_mm):
    """Dodaj skalu"""
    if image is None:
        return image
        
    h, w = image.shape[:2]
    mm_per_pixel = hvid_mm / (image.shape[1] // 3)  # jer je centralna 1/3 širine
    
    scale_length_px = int(5 / mm_per_pixel)
    
    margin = 30
    y_start = h - margin - 10
    y_end = h - margin
    x_start = margin
    x_end = x_start + scale_length_px
    
    cv2.rectangle(image, (x_start-2, y_start-2), (x_end+2, y_end+2), (255, 255, 255), -1)
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
    cv2.putText(image, "5 mm", (x_start, y_start-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def add_labels(image, eye_side):
    """Dodaj oznake"""
    if image is None:
        return image
        
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Oznake za kvadrante
    if eye_side == "Left eye (OS)":
        cv2.putText(image, "NASAL", (20, h//2), font, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "TEMPORAL", (w-100, h//2), font, 0.6, (0, 0, 0), 2)
    else:
        cv2.putText(image, "NASAL", (20, h//2), font, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "TEMPORAL", (w-100, h//2), font, 0.6, (0, 0, 0), 2)
    
    cv2.putText(image, "SUPERIOR", (w//2-40, 30), font, 0.6, (0, 0, 0), 2)
    cv2.putText(image, "INFERIOR", (w//2-40, h-30), font, 0.6, (0, 0, 0), 2)
    
    return image

# Streamlit UI
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Settings")
    
    eye_side = st.radio(
        "Select eye",
        ["Left eye (OS)", "Right eye (OD)"],
        index=0
    )
    
    hvid_input = st.number_input("HVID (mm)", 
                                 min_value=10.0, max_value=14.0, value=11.5, step=0.1)
    
    st.markdown("---")
    st.header("Upload Images")
    
    with st.expander("📖 Instructions", expanded=False):
        st.markdown("""
        **Upload 5 images:**
        - **Central fix** - straight ahead
        - **Up fix** - patient looks UP (images INFERIOR)
        - **Down fix** - patient looks DOWN (images SUPERIOR)
        - **Left fix** - patient looks LEFT
        - **Right fix** - patient looks RIGHT
        """)
    
    central = st.file_uploader("Central fix", type=['jpg', 'jpeg', 'png'], key='central')
    up = st.file_uploader("Up fix (looks up)", type=['jpg', 'jpeg', 'png'], key='up')
    down = st.file_uploader("Down fix (looks down)", type=['jpg', 'jpeg', 'png'], key='down')
    left = st.file_uploader("Left fix (looks left)", type=['jpg', 'jpeg', 'png'], key='left')
    right = st.file_uploader("Right fix (looks right)", type=['jpg', 'jpeg', 'png'], key='right')
    
    process = st.button("🔄 Create Wide Panorama", type="primary", use_container_width=True)

with col2:
    st.header("Result")
    
    if process:
        if central is None:
            st.error("⚠️ Please upload at least the central image!")
        else:
            with st.spinner("🔄 Creating wide corneal panorama..."):
                # Učitaj slike
                images = {}
                files = {
                    'Central fix': central,
                    'Up fix': up,
                    'Down fix': down,
                    'Left fix': left,
                    'Right fix': right
                }
                
                for name, file in files.items():
                    if file:
                        img = load_image(file)
                        if img is not None:
                            cropped = crop_to_circle(img)
                            if cropped is not None:
                                images[name] = cropped
                
                if 'Central fix' in images:
                    # Napravi panoramu
                    panorama = create_wide_panorama(images, eye_side)
                    
                    if panorama is not None:
                        # Dodaj oznake i skalu
                        panorama = add_labels(panorama, eye_side)
                        panorama = add_scale_bar(panorama, hvid_input)
                        
                        st.image(panorama, caption="Wide Corneal Panorama", use_container_width=True)
                        
                        # Download
                        panorama_bgr = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
                        _, buffer = cv2.imencode('.png', panorama_bgr)
                        
                        eye_code = "OS" if eye_side == "Left eye (OS)" else "OD"
                        st.download_button(
                            label="📥 Download Panorama",
                            data=buffer.tobytes(),
                            file_name=f"topostitcher_panorama_{eye_code}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                        used = [n.replace(' fix', '') for n in images.keys()]
                        st.success(f"✅ Success! Used: {', '.join(used)}")
                    else:
                        st.error("❌ Could not create panorama!")
                else:
                    st.error("❌ Central image required!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
<b>TopoStitcher</b> - Creates ONE wide image from 5 corneal topography images
</div>
""", unsafe_allow_html=True)
