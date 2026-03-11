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

def crop_to_circle(image):
    """Izreži samo kružni dio topografske mape"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Napravi masku s mekim rubom za bolje spajanje
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Dodaj mali blur na rub za glatko spajanje
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Izreži na bounding box kruga
        y1 = max(0, center[1] - radius)
        y2 = min(image.shape[0], center[1] + radius)
        x1 = max(0, center[0] - radius)
        x2 = min(image.shape[1], center[0] + radius)
        
        return result[y1:y2, x1:x2], (center, radius)
    
    return None, None

def find_overlap_and_stitch(img1, img2, direction):
    """Pronađi preklapanje i spoji dvije slike"""
    # Pretvori u grayscale za detekciju značajki
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Detekcija značajki (krvne žile, rubovi, uzorci)
    sift = cv2.SIFT_create()
    
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    if descriptors1 is None or descriptors2 is None:
        return None
    
    # FLANN matcher za brzo prepoznavanje
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Lowe's ratio test za dobra preklapanja
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        # Ako nema dovoljno preklapanja, pokušaj jednostavniji pristup
        return simple_stitch(img1, img2, direction)
    
    # Pronađi homografiju
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        return simple_stitch(img1, img2, direction)
    
    # Spoji slike
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Transformiraj prvu sliku u prostor druge
    result = cv2.warpPerspective(img1, H, (w1 + w2, max(h1, h2)))
    result[0:h2, 0:w2] = img2
    
    # Izreži prazan prostor
    gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    non_empty = np.where(gray_result > 0)
    if len(non_empty[0]) > 0:
        y_min, y_max = np.min(non_empty[0]), np.max(non_empty[0])
        x_min, x_max = np.min(non_empty[1]), np.max(non_empty[1])
        result = result[y_min:y_max+1, x_min:x_max+1]
    
    return result

def simple_stitch(img1, img2, direction):
    """Fallback metoda ako feature matching ne radi"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Pronađi preklapanje pomoću korelacije
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Ograniči pretragu na očekivano područje preklapanja
    if direction in ['left', 'right']:
        search_width = min(w1, w2) // 3
        if direction == 'right':  # img2 je desno od img1
            template = gray1[:, -search_width:]
            result = cv2.matchTemplate(gray2[:, :search_width*2], template, cv2.TM_CCOEFF_NORMED)
        else:  # img2 je lijevo od img1
            template = gray1[:, :search_width]
            result = cv2.matchTemplate(gray2[:, -search_width*2:], template, cv2.TM_CCOEFF_NORMED)
    else:  # gore/dolje
        search_height = min(h1, h2) // 3
        if direction == 'down':  # img2 je dolje od img1
            template = gray1[-search_height:, :]
            result = cv2.matchTemplate(gray2[:search_height*2, :], template, cv2.TM_CCOEFF_NORMED)
        else:  # img2 je gore od img1
            template = gray1[:search_height, :]
            result = cv2.matchTemplate(gray2[-search_height*2:, :], template, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val < 0.3:  # Ako nema dobrog preklapanja
        # Jednostavno stavi jednu pored druge
        if direction in ['left', 'right']:
            result_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            result_img[:h1, :w1] = img1
            if direction == 'right':
                result_img[:h2, w1:w1+w2] = img2
            else:
                result_img[:h2, :w2] = img2
                result_img[:h1, w2:w2+w1] = img1
        else:
            result_img = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
            result_img[:h1, :w1] = img1
            if direction == 'down':
                result_img[h1:h1+h2, :w2] = img2
            else:
                result_img[:h2, :w2] = img2
                result_img[h2:h2+h1, :w1] = img1
    else:
        # Spoji s pronađenim preklapanjem
        if direction in ['left', 'right']:
            overlap = max_loc[0] if direction == 'right' else w2 - max_loc[0]
            result_img = np.zeros((max(h1, h2), w1 + w2 - overlap, 3), dtype=np.uint8)
            # Linearno blendanje u zoni preklapanja
            # (pojednostavljeno za sada)
            if direction == 'right':
                result_img[:h1, :w1] = img1
                result_img[:h2, w1-overlap:w1-overlap+w2] = img2
            else:
                result_img[:h2, :w2] = img2
                result_img[:h1, w2-overlap:w2-overlap+w1] = img1
        else:
            overlap = max_loc[1] if direction == 'down' else h2 - max_loc[1]
            result_img = np.zeros((h1 + h2 - overlap, max(w1, w2), 3), dtype=np.uint8)
            if direction == 'down':
                result_img[:h1, :w1] = img1
                result_img[h1-overlap:h1-overlap+h2, :w2] = img2
            else:
                result_img[:h2, :w2] = img2
                result_img[h2-overlap:h2-overlap+h1, :w1] = img1
    
    return result_img

def add_scale_bar(image, hvid_mm):
    """Dodaj skalu na osnovu HVID-a"""
    h, w = image.shape[:2]
    mm_per_pixel = hvid_mm / w
    
    # Skala od 2 mm (stavi na dno)
    scale_length_px = int(2 / mm_per_pixel)
    
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
    scale_text = f"2 mm"
    cv2.putText(image, scale_text, (x_start, y_start-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def add_quadrant_labels(image):
    """Dodaj oznake kvadranata"""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 0, 0)
    bg_color = (255, 255, 255)
    
    margin = 40
    
    # Pozicije
    labels = {
        'S': (w//2, margin),      # Superior (gore)
        'I': (w//2, h - margin),   # Inferior (dolje)
        'N': (margin, h//2),       # Nasal (lijevo)
        'T': (w - margin, h//2)     # Temporal (desno)
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

def create_panorama(images_dict, is_left_eye):
    """Kreiraj panoramu postupnim spajanjem"""
    if 'Central fix' not in images_dict:
        return None
    
    # Počni sa centralnom
    panorama = images_dict['Central fix'].copy()
    h, w = panorama.shape[:2]
    
    # Definiraj redoslijed spajanja i smjerove
    if is_left_eye:
        order = [
            ('Right fix', 'left'),   # nazalno dolazi s lijeve strane
            ('Left fix', 'right'),    # temporalno s desne
            ('Up fix', 'up'),         # gore
            ('Down fix', 'down')       # dolje
        ]
    else:
        order = [
            ('Left fix', 'left'),     # za desno oko, nazalno je lijevo
            ('Right fix', 'right'),    # temporalno desno
            ('Up fix', 'up'),
            ('Down fix', 'down')
        ]
    
    status_messages = []
    
    for img_name, direction in order:
        if img_name in images_dict:
            img = images_dict[img_name]
            status_messages.append(f"Spajam {img_name} ({direction})")
            
            # Spoji prema smjeru
            if direction == 'left':
                # Slika treba biti lijevo od panorame
                stitched = find_overlap_and_stitch(img, panorama, 'right')
                if stitched is not None:
                    panorama = stitched
            elif direction == 'right':
                # Slika treba biti desno od panorame
                stitched = find_overlap_and_stitch(panorama, img, 'right')
                if stitched is not None:
                    panorama = stitched
            elif direction == 'up':
                # Slika treba biti gore
                stitched = find_overlap_and_stitch(img, panorama, 'down')
                if stitched is not None:
                    panorama = stitched
            elif direction == 'down':
                # Slika treba biti dolje
                stitched = find_overlap_and_stitch(panorama, img, 'down')
                if stitched is not None:
                    panorama = stitched
    
    return panorama, status_messages

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
        - **Central fix** - straight ahead (central map)
        - **Up fix** - patient looks **UP** (images **INFERIOR**)
        - **Down fix** - patient looks **DOWN** (images **SUPERIOR**)  
        - **Left fix** - patient looks **LEFT**
        - **Right fix** - patient looks **RIGHT**
        
        The app will automatically detect overlapping areas based on blood vessels and pattern matching.
        """)
    
    central = st.file_uploader("Central fix", type=['jpg', 'jpeg', 'png'], key='central')
    up = st.file_uploader("Up fix (looks up)", type=['jpg', 'jpeg', 'png'], key='up')
    down = st.file_uploader("Down fix (looks down)", type=['jpg', 'jpeg', 'png'], key='down')
    left = st.file_uploader("Left fix (looks left)", type=['jpg', 'jpeg', 'png'], key='left')
    right = st.file_uploader("Right fix (looks right)", type=['jpg', 'jpeg', 'png'], key='right')
    
    process = st.button("🔄 Create Peripheral Panorama", type="primary", use_container_width=True)

with col2:
    st.header("Result")
    
    if process:
        if central is None:
            st.error("⚠️ Please upload at least the central image!")
        else:
            with st.spinner("🔄 Processing images and detecting overlap..."):
                # Učitaj sve slike i izreži krugove
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
                            cropped, _ = crop_to_circle(img_rgb)
                            if cropped is not None:
                                images[name] = cropped
                    progress.progress((i + 1) / len(files))
                
                status.text("Creating panorama with feature matching...")
                
                if len(images) >= 2:
                    # Kreiraj panoramu
                    panorama, stitch_status = create_panorama(images, is_left_eye)
                    
                    if panorama is not None:
                        # Dodaj skalu i oznake
                        panorama_with_labels = add_quadrant_labels(panorama.copy())
                        panorama_with_scale = add_scale_bar(panorama_with_labels, hvid_input)
                        
                        st.image(panorama_with_scale, caption="Peripheral Corneal Panorama", use_container_width=True)
                        
                        # Download
                        _, buffer = cv2.imencode('.png', cv2.cvtColor(panorama_with_scale, cv2.COLOR_RGB2BGR))
                        st.download_button(
                            label="📥 Download Panorama",
                            data=buffer.tobytes(),
                            file_name=f"topostitcher_panorama_{'OS' if is_left_eye else 'OD'}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                        
                        # Status
                        used = [n.replace(' fix', '') for n in images.keys()]
                        st.success(f"✅ Success! Stitched: {', '.join(used)}")
                        
                        with st.expander("📊 Stitching Details"):
                            for msg in stitch_status:
                                st.write(f"• {msg}")
                    else:
                        st.error("❌ Could not create panorama!")
                else:
                    st.error("❌ Need at least 2 images for stitching!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
<b>TopoStitcher</b> v1.1 | For orientation purposes only<br>
Uses SIFT feature matching to detect overlapping blood vessels and patterns
</div>
""", unsafe_allow_html=True)
