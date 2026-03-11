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
    """Pouzdano učitavanje slike iz uploadanog fajla"""
    if uploaded_file is not None:
        # Pročitaj bytes
        bytes_data = uploaded_file.read()
        if len(bytes_data) == 0:
            return None
        
        # Konvertiraj u numpy array
        nparr = np.frombuffer(bytes_data, np.uint8)
        
        # Dekodiraj sliku
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            # Konvertiraj BGR u RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
    
    return None

def crop_to_circle(image):
    """Izreži samo kružni dio topografske mape"""
    if image is None:
        return None, None
        
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
        
        return result[y1:y2, x1:x2], (center, radius)
    
    return None, None

def add_scale_bar(image, hvid_mm):
    """Dodaj skalu na osnovu HVID-a"""
    if image is None:
        return image
        
    h, w = image.shape[:2]
    mm_per_pixel = hvid_mm / w
    
    # Skala od 2 mm
    scale_length_px = int(2 / mm_per_pixel)
    if scale_length_px < 20:  # Ako je 2mm premalo, koristi 5mm
        scale_length_px = int(5 / mm_per_pixel)
        scale_text = "5 mm"
    else:
        scale_text = "2 mm"
    
    margin = 30
    y_start = h - margin - 10
    y_end = h - margin
    x_start = margin
    x_end = x_start + scale_length_px
    
    # Bijela pozadina
    cv2.rectangle(image, (x_start-2, y_start-2), (x_end+2, y_end+2), (255, 255, 255), -1)
    # Crna skala
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
    
    # Tekst
    cv2.putText(image, scale_text, (x_start, y_start-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def add_quadrant_labels(image):
    """Dodaj oznake kvadranata"""
    if image is None:
        return image
        
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 0, 0)
    bg_color = (255, 255, 255)
    
    margin = 40
    
    labels = {
        'S': (w//2, margin),
        'I': (w//2, h - margin),
        'N': (margin, h//2),
        'T': (w - margin, h//2)
    }
    
    for label, (x, y) in labels.items():
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        # Pozadina
        cv2.rectangle(image, (x - text_w//2 - 4, y - text_h//2 - 4), 
                     (x + text_w//2 + 4, y + text_h//2 + 4), bg_color, -1)
        # Tekst
        cv2.putText(image, label, (x - text_w//2, y + text_h//2), 
                   font, font_scale, color, thickness)
    
    return image

def create_simple_mosaic(images_dict, eye_side):
    """Napravi jednostavan mozaik od dostupnih slika"""
    if 'Central fix' not in images_dict:
        return None
    
    central = images_dict['Central fix']
    h, w = central.shape[:2]
    
    # Definiraj pozicije
    positions = {
        'Central fix': (h, w),  # centar
    }
    
    if eye_side == "Left eye (OS)":
        # Za lijevo oko
        if 'Right fix' in images_dict:  # nazalno (lijevo)
            positions['Right fix'] = (h, 0)
        if 'Left fix' in images_dict:   # temporalno (desno)
            positions['Left fix'] = (h, 2*w)
        if 'Up fix' in images_dict:      # gore (superior) - zapravo snima dolje
            positions['Up fix'] = (0, w)
        if 'Down fix' in images_dict:    # dolje (inferior) - zapravo snima gore
            positions['Down fix'] = (2*h, w)
    else:
        # Za desno oko
        if 'Left fix' in images_dict:    # nazalno (lijevo)
            positions['Left fix'] = (h, 0)
        if 'Right fix' in images_dict:   # temporalno (desno)
            positions['Right fix'] = (h, 2*w)
        if 'Up fix' in images_dict:       # gore
            positions['Up fix'] = (0, w)
        if 'Down fix' in images_dict:     # dolje
            positions['Down fix'] = (2*h, w)
    
    # Napravi canvas
    canvas = np.ones((h*3, w*3, 3), dtype=np.uint8) * 255
    
    # Postavi centralnu
    canvas[h:h*2, w:w*2] = central
    
    # Postavi ostale
    for name, (y, x) in positions.items():
        if name != 'Central fix' and name in images_dict:
            img = images_dict[name]
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            canvas[y:y+h, x:x+w] = img
    
    # Izreži prazan prostor
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    non_empty = np.where(gray < 254)  # nije bijelo
    if len(non_empty[0]) > 0:
        y_min, y_max = np.min(non_empty[0]), np.max(non_empty[0])
        x_min, x_max = np.min(non_empty[1]), np.max(non_empty[1])
        canvas = canvas[y_min:y_max+1, x_min:x_max+1]
    
    return canvas

# Streamlit UI
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Settings")
    
    # Izbor oka - sada radi za oba!
    eye_side = st.radio(
        "Select eye",
        ["Left eye (OS)", "Right eye (OD)"],
        index=0,
        help="Choose left or right eye - orientation will be adjusted automatically"
    )
    
    hvid_input = st.number_input("HVID (mm)", 
                                 min_value=10.0, max_value=14.0, value=11.5, step=0.1,
                                 help="Horizontal Visible Iris Diameter in mm")
    
    st.markdown("---")
    st.header("Upload Images")
    
    with st.expander("📖 Instructions", expanded=False):
        st.markdown("""
        **Fixation guide:**
        - **Central fix** - straight ahead
        - **Up fix** - patient looks **UP** (images **INFERIOR** quadrant)
        - **Down fix** - patient looks **DOWN** (images **SUPERIOR** quadrant)  
        - **Left fix** - patient looks **LEFT**
        - **Right fix** - patient looks **RIGHT**
        
        The app automatically adjusts orientation for left/right eye.
        """)
    
    # Upload fajlova - koristimo unique key za svaki
    central_file = st.file_uploader("Central fix", type=['jpg', 'jpeg', 'png'], key='upload_central')
    up_file = st.file_uploader("Up fix (looks up)", type=['jpg', 'jpeg', 'png'], key='upload_up')
    down_file = st.file_uploader("Down fix (looks down)", type=['jpg', 'jpeg', 'png'], key='upload_down')
    left_file = st.file_uploader("Left fix (looks left)", type=['jpg', 'jpeg', 'png'], key='upload_left')
    right_file = st.file_uploader("Right fix (looks right)", type=['jpg', 'jpeg', 'png'], key='upload_right')
    
    process = st.button("🔄 Create Mosaic", type="primary", use_container_width=True)
    
    # Debug info
    with st.expander("🔧 Debug Info"):
        st.write("Upload status:")
        st.write(f"Central: {'✅' if central_file else '❌'}")
        st.write(f"Up: {'✅' if up_file else '❌'}")
        st.write(f"Down: {'✅' if down_file else '❌'}")
        st.write(f"Left: {'✅' if left_file else '❌'}")
        st.write(f"Right: {'✅' if right_file else '❌'}")

with col2:
    st.header("Result")
    
    if process:
        if central_file is None:
            st.error("⚠️ Please upload at least the central image!")
        else:
            with st.spinner("🔄 Processing images..."):
                # Učitaj sve slike
                images = {}
                files = {
                    'Central fix': central_file,
                    'Up fix': up_file,
                    'Down fix': down_file,
                    'Left fix': left_file,
                    'Right fix': right_file
                }
                
                progress = st.progress(0)
                status = st.empty()
                
                for i, (name, file) in enumerate(files.items()):
                    status.text(f"Processing {name}...")
                    if file:
                        img = load_image(file)
                        if img is not None:
                            cropped, _ = crop_to_circle(img)
                            if cropped is not None:
                                images[name] = cropped
                                st.write(f"✅ {name} loaded: {cropped.shape}")
                            else:
                                st.write(f"⚠️ {name} - could not crop circle")
                        else:
                            st.write(f"⚠️ {name} - could not load image")
                    progress.progress((i + 1) / len(files))
                
                status.text("Creating mosaic...")
                
                if len(images) >= 1:
                    # Napravi mozaik
                    mosaic = create_simple_mosaic(images, eye_side)
                    
                    if mosaic is not None:
                        # Dodaj oznake
                        mosaic_with_labels = add_quadrant_labels(mosaic.copy())
                        mosaic_with_scale = add_scale_bar(mosaic_with_labels, hvid_input)
                        
                        # Prikaži
                        st.image(mosaic_with_scale, caption="Corneal Topography Mosaic", use_container_width=True)
                        
                        # Download
                        mosaic_bgr = cv2.cvtColor(mosaic_with_scale, cv2.COLOR_RGB2BGR)
                        _, buffer = cv2.imencode('.png', mosaic_bgr)
                        
                        eye_code = "OS" if eye_side == "Left eye (OS)" else "OD"
                        st.download_button(
                            label="📥 Download Mosaic",
                            data=buffer.tobytes(),
                            file_name=f"topostitcher_{eye_code}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                        # Koje slike su korištene
                        used = [n.replace(' fix', '') for n in images.keys()]
                        st.success(f"✅ Success! Used: {', '.join(used)}")
                    else:
                        st.error("❌ Could not create mosaic!")
                else:
                    st.error("❌ Could not load any images!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
<b>TopoStitcher</b> v1.2 | For orientation purposes only<br>
© Phantasmed
</div>
""", unsafe_allow_html=True)
