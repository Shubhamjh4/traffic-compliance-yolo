import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# 1. Page Configuration
st.set_page_config(
    page_title="Culturally-Aware Traffic Compliance Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load Your YOLOv8 Model
# Ensure your custom trained 'best.pt' model file is in the same directory
@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

try:
    model = load_model()
except Exception as e:
    st.error("Could not load 'best.pt' model. Please ensure the weights file is in the script directory.")
    model = None

# 3. Sidebar: Research Highlights & Metric Proofs
with st.sidebar:
    st.header("Research Highlights")
    st.markdown("""
    **The Problem:**
    Standard traffic AI cameras use binary classification (Helmet/No-Helmet). 
    Consequently, they falsely penalize Sikh riders wearing Turbans, violating 
    **Section 129 of the Indian Motor Vehicles Act** and generating massive administrative overhead.
    
    **The Solution:**
    A multi-class object detection framework that explicitly distinguishes between 
    *Helmet*, *No-Headgear*, and *Turban* classes, embedding statutory law into computer vision.
    """)
    
    st.subheader("Model Performance Proofs")
    # Place your metric image file names here (e.g., 'confusion_matrix.png' or 'results.png')
    try:
        st.image("results_graph.png", caption="Training Metrics & Curves", use_container_width=True)
    except:
        st.caption("[Insert your metrics_summary.png here]")
        
    st.markdown("""
    - **Architecture:** YOLOv8n (Nano)
    - **Peak mAP@50:** 88.17% (Epoch 38)
    - **Precision:** 89.59%
    - **Recall:** 82.26%
    """)

# 4. Main Application Interface
st.title("Automated Intelligent Traffic Compliance Platform")
st.subheader("Upload an Image or Video to Test Real-Time Compliance Inference")

if model is not None:
    # Media selector tab
    file_type = st.radio("Select input type:", ("Image", "Video"))
    
    if file_type == "Image":
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            # Run Inference
            # confidence threshold set to 0.5 matching NMS logic
            results = model.predict(source=opencv_image, conf=0.5) 
            
            # Plot bounding boxes on image
            annotated_img = results[0].plot()
            
            # Convert BGR back to RGB for streamlit display
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_img_rgb, caption="Inference Result (Statutory Logic Applied)", use_container_width=True)

    elif file_type == "Video":
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            # Save uploaded video to a temporary file to read via OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            
            video_cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            st.info("Processing video frames... results streaming below.")
            
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break
                
                # Run YOLOv8 model prediction frame by frame
                results = model.predict(source=frame, conf=0.5)
                annotated_frame = results[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Stream the frame to web page
                st_frame.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
            video_cap.release()
            st.success("Video processing complete.")

# 5. Persistent Footer Highlighting the Research Context
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    <p><b>Research Framework:</b> Culturally-Aware Automated Traffic Compliance Framework via YOLOv8n</p>
    <p>Unlike traditional binary classifiers, this architecture implements rule-based statutory integration mapping predictions directly to compliance states (Safe vs. Violation vs. Section 129 Legal Turban Exemption).</p>
</div>
""", unsafe_allow_html=True)