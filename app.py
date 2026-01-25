"""
Recyclable Object Detection - Streamlit App (Hugging Face Spaces Version)
==========================================================================

YOLOv8s model untuk deteksi objek daur ulang.

Performance:
  - mAP@0.5: 63.4%
  - Precision: 71.8%
  - Recall: 70.0%
  - F1-Score: 0.709
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# Page config
st.set_page_config(
    page_title="Recyclable Object Detection",
    page_icon="♻️",
    layout="wide"
)

# Constants
MODEL_PATH = "best.pt"
DEFAULT_CONFIDENCE = 0.20

# Cache model loading
@st.cache_resource
def load_model():
    """Load YOLO model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

def draw_boxes(image, results, conf_threshold):
    """Draw bounding boxes on image"""
    img = np.array(image)
    # Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf >= conf_threshold:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw box
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"recyclable {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_bgr, (x1, y1 - 25), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def main():
    # Header
    st.title("♻️ Recyclable Object Detection")
    st.markdown("**YOLOv8s Model** - Deteksi objek yang dapat didaur ulang")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.05,
        help="Threshold untuk menampilkan deteksi. Nilai lebih rendah = lebih banyak deteksi."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.markdown("""
    | Metric | Value |
    |--------|-------|
    | mAP@0.5 | 63.4% |
    | Precision | 71.8% |
    | Recall | 70.0% |
    | F1-Score | 0.709 |
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"📝 Confidence threshold: **{confidence:.2f}**")
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Pilih gambar untuk dideteksi",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Format yang didukung: JPG, JPEG, PNG, WEBP"
        )
        
        # Sample images option
        use_sample = st.checkbox("🖼️ Gunakan gambar sampel")
        
        sample_path = None
        if use_sample:
            sample_dir = "samples"
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if sample_files:
                    selected_sample = st.selectbox("Pilih gambar sampel:", sample_files)
                    sample_path = os.path.join(sample_dir, selected_sample)
    
    # Process image
    if uploaded_file is not None or sample_path is not None:
        # Load image
        if sample_path is not None:
            image = Image.open(sample_path).convert('RGB')
        else:
            image = Image.open(uploaded_file).convert('RGB')
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        # Run detection
        with st.spinner("🔍 Detecting objects..."):
            img_array = np.array(image)
            results = model.predict(source=img_array, conf=0.001, verbose=False)
        
        # Draw results
        result_image = draw_boxes(image, results, confidence)
        
        with col2:
            st.subheader("🎯 Detection Results")
            st.image(result_image, caption="Detected Objects", use_container_width=True)
        
        # Statistics
        if results and len(results) > 0:
            boxes = results[0].boxes
            total_detections = sum(1 for box in boxes if float(box.conf[0]) >= confidence)
            all_confs = [float(box.conf[0]) for box in boxes if float(box.conf[0]) >= confidence]
            
            st.markdown("---")
            st.subheader("📈 Detection Statistics")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("Total Detections", total_detections)
            
            with stat_col2:
                if all_confs:
                    st.metric("Avg Confidence", f"{np.mean(all_confs):.2%}")
                else:
                    st.metric("Avg Confidence", "N/A")
            
            with stat_col3:
                if all_confs:
                    st.metric("Max Confidence", f"{max(all_confs):.2%}")
                else:
                    st.metric("Max Confidence", "N/A")
            
            # Detailed detections table
            if total_detections > 0:
                st.markdown("### 📋 Detection Details")
                detection_data = []
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    if conf >= confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detection_data.append({
                            "No": len(detection_data) + 1,
                            "Class": "recyclable",
                            "Confidence": f"{conf:.2%}",
                            "Box (x1,y1,x2,y2)": f"({x1}, {y1}, {x2}, {y2})"
                        })
                
                st.dataframe(detection_data, use_container_width=True)
            else:
                st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
    
    else:
        # Welcome message
        st.info("👆 Upload an image to start detection.")
        
        # Performance comparison
        st.markdown("---")
        st.subheader("📊 Model Performance: Before vs After Training")
        
        before_col, after_col = st.columns(2)
        
        with before_col:
            st.markdown("### ❌ Before (Original)")
            st.markdown("""
            | Metric | Value |
            |--------|-------|
            | mAP@0.5 | 43.7% |
            | Precision | 57.9% |
            | Recall | 55.1% |
            | F1-Score | 0.565 |
            """)
        
        with after_col:
            st.markdown("### ✅ After (Retrained)")
            st.markdown("""
            | Metric | Value |
            |--------|-------|
            | mAP@0.5 | **63.4%** |
            | Precision | **71.8%** |
            | Recall | **70.0%** |
            | F1-Score | **0.709** |
            """)
        
        st.success("**Improvement**: mAP +19.7%, Precision +13.9%, Recall +14.9%")


if __name__ == "__main__":
    main()
