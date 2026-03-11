import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Konfiguracija stranice - MORA BITI PRVA STREAMLIT KOMANDA
st.set_page_config(
    page_title="TopoStitcher - Corneal Topography Mosaicing",
    page_icon="🩺",
    layout="wide"
)

# Učitaj logo - jednostavno, bez try/except
logo_path = Path(__file__).parent / "Phantasmed-logo.png"
if logo_path.exists():
    logo = Image.open(logo_path)
else:
    logo = None

# Header s logom
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

# Sve ostalo je isto kao u prethodnoj verziji, ali ću ponoviti cijeli kod radi kompletnosti

def crop_to_circle(image):
    """Crop only the circular topographic map"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        result = cv2.bitwise_and(image, image, mask=mask)
        
        rgba = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = mask
        
        y1 = max(0, center[1] - radius)
        y2 = min(image.shape[0], center[1] + radius)
        x1 = max(0, center[0] - radius)
        x2 = min(image.shape[1], center[0] + radius)
        
        return rgba[y1:y2, x1:x2]
    
    return None

def add_scale_bar(image, hvid_mm):
    """Add scale bar based on HVID"""
    h, w = image.shape[:2]
    
    mm_per_pixel = hvid_mm / w
    scale_length_px = int(5 / mm_per_pixel)
    if scale_length_px > w // 4:
        scale_length_px = int(2 / mm_per_pixel)
    
    margin = 20
    y_start = h - margin - 10
    y_end = h - margin
    x_start = margin
    x_end = x_start + scale_length_px
    
    cv2.rectangle(image, (x_start-2, y_start-2), (x_end+2, y_end+2), (255, 255, 255), -1)
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)
    
    scale_text = f"{scale_length_px * mm_per_pixel:.1f} mm"
    cv2.putText(image, scale_text, (x_start, y_start-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def add_quadrant_labels(image):
    """Add quadrant labels (S, I, N, T)"""
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
        cv2.rectangle(image, (x - text_w//2 - 4, y - text_h//2 - 4), 
                     (x + text_w//2 + 4, y + text_h//2 + 4), bg_color, -1)
        cv2.putText(image, label, (x - text_w//2, y + text_h//2), 
                   font, font_scale, color, thickness)
    
    return image

def stitch_images(images_dict, is_left_eye, hvid_value):
    """Stitch images into mosaic"""
    if not images_dict or 'Central fix' not in images_dict:
        return None
    
    central = images_dict['Central fix']
    h, w = central.shape[:2]
    
    POSITIONS = {
        'Central fix': (1, 1),
        'Up fix': (0, 1),
        'Down fix': (2, 1),
        'Left fix': (1, 2),
        'Right fix': (1, 0)
    }
    
    OD_FLIP = {
        'Left fix': (1, 0),
        'Right fix': (1, 2)
    }
    
    canvas = np.zeros((h*3, w*3, 4), dtype=np.uint8)
    canvas[h:h*2, w:w*2] = central
    
    for name, img in images_dict.items():
        if name == 'Central fix' or img is None:
            continue
        
        if is_left_eye:
            pos = POSITIONS.get(name)
        else:
            pos = OD_FLIP.get(name, POSITIONS.get(name))
        
        if pos:
            row, col = pos
            y = row * h
            x = col * w
            
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            
            alpha = img[:, :, 3] / 255.0
            for c in range(3):
                canvas[y:y+h, x:x+w, c] = (1 - alpha) * canvas[y:y+h, x:x+w, c] + alpha * img[:, :, c]
            canvas[y:y+h, x:x+w, 3] = np.maximum(canvas[y:y+h, x:x+w, 3], img[:, :, 3])
    
    non_empty = np.where(canvas[:, :, 3] > 0)
    if len(non_empty[0]) > 0:
        y_min, y_max = np.min(non_empty[0]), np.max(non_empty[0])
        x_min, x_max = np.min(non_empty[1]), np.max(non_empty[1])
        canvas = canvas[y_min:y_max+1, x_min:x_max+1]
    
    result = cv2.cvtColor(canvas[:, :, :3], cv2.COLOR_RGB2BGR)
    
    if hvid_value > 0:
        result = add_scale_bar(result, hvid_value)
    
    result = add_quadrant_labels(result)
    
    return result

# Streamlit UI
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Settings")
    
    is_left_eye = st.checkbox("Left eye (OS)", value=True, 
                              help="Check for left eye (OS), uncheck for right eye (OD)")
    
    hvid_input = st.number_input("HVID (mm)", 
                                 min_value=10.0, max_value=14.0, value=11.5, step=0.1,
                                 help="Horizontal Visible Iris Diameter in mm")
    
    st.markdown("---")
    st.header("Upload Images")
    
    with st.expander("📖 Instructions", expanded=False):
        st.markdown("""
        **Fixation guide:**
        - **Central fix** - straight ahead
        - **Up fix** - patient looks **UP** (images **INFERIOR**)
        - **Down fix** - patient looks **DOWN** (images **SUPERIOR**)  
        - **Left fix** - patient looks **LEFT**
        - **Right fix** - patient looks **RIGHT**
        
        App automatically handles left/right eye orientation.
        """)
    
    central = st.file_uploader("Central fix", type=['jpg', 'jpeg', 'png'], key='central')
    up = st.file_uploader("Up fix (looks up)", type=['jpg', 'jpeg', 'png'], key='up')
    down = st.file_uploader("Down fix (looks down)", type=['jpg', 'jpeg', 'png'], key='down')
    left = st.file_uploader("Left fix (looks left)", type=['jpg', 'jpeg', 'png'], key='left')
    right = st.file_uploader("Right fix (looks right)", type=['jpg', 'jpeg', 'png'], key='right')
    
    process = st.button("🔄 Stitch Images", type="primary", use_container_width=True)

with col2:
    st.header("Result")
    
    if process:
        if central is None:
            st.error("⚠️ Please upload at least the central image!")
        else:
            with st.spinner("🔄 Processing images..."):
                images = {}
                files = {
                    'Central fix': central,
                    'Up fix': up,
                    'Down fix': down,
                    'Left fix': left,
                    'Right fix': right
                }
                
                progress = st.progress(0)
                status = st.empty()
                
                for i, (name, file) in enumerate(files.items()):
                    status.text(f"Processing {name}...")
                    if file:
                        bytes_data = file.read()
                        nparr = np.frombuffer(bytes_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            cropped = crop_to_circle(img_rgb)
                            if cropped is not None:
                                images[name] = cropped
                    progress.progress((i + 1) / len(files))
                
                status.text("Stitching images...")
                
                if images:
                    result = stitch_images(images, is_left_eye, hvid_input)
                    
                    if result is not None:
                        st.image(result, caption="Stitched Topography Map", use_container_width=True)
                        
                        _, buffer = cv2.imencode('.png', result)
                        st.download_button(
                            label="📥 Download Stitched Image",
                            data=buffer.tobytes(),
                            file_name=f"topostitcher_{'OS' if is_left_eye else 'OD'}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                        used = [n.replace(' fix', '') for n in images.keys()]
                        st.success(f"✅ Success! Used: {', '.join(used)}")
                    else:
                        st.error("❌ Error stitching images!")
                else:
                    st.error("❌ Could not load any images!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
<b>TopoStitcher</b> v1.0 | For orientation purposes only<br>
Not a clinical standard - does not replace dedicated scleral topographers
</div>
""", unsafe_allow_html=True)