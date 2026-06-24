import streamlit as st
import cv2
import requests
import base64
import numpy as np
import pandas as pd
import math
from PIL import Image
from streamlit_cropper import st_cropper

# --- 1. YOUR ROBOFLOW SETTINGS ---
# 🚨 UPDATE YOUR PROJECT ID TO YOUR NEW 96.1% VERSION
API_KEY = "6t3RCD7zG6anysk3AjGv"
PROJECT_ID = "fuel-storage-tanks-i8qsn/1"

st.set_page_config(layout="wide", page_title="StorageTanks")

# --- 2. THE SIDEBAR (Calibration & Tools) ---
st.sidebar.title("⚙️ Operations Menu")

gsd_cm = st.sidebar.number_input("Ground Sample Distance (cm/px)", min_value=1.0, value=5.0, step=0.5)
tank_height_m = st.sidebar.number_input("Estimated Tank Height (meters)", min_value=1.0, value=10.0, step=1.0)
confidence_threshold = st.sidebar.slider("AI Confidence Threshold (%)", min_value=1, max_value=100, value=40, step=5)

gsd_m = gsd_cm / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Advanced Tools")
# The toggle switch for our new Magnifying Glass!
enable_crop = st.sidebar.checkbox("Enable Magnifying Glass (Crop Mode)")

# --- 3. MAIN DASHBOARD ---
st.title("🌍FuelStorage Tank Detector")
st.markdown("Upload high-resolution aerial imagery. Use the Magnifying Glass to target micro-infrastructure.")

uploaded_file = st.file_uploader("Upload Imagery (JPG/PNG)", type=['jpg', 'jpeg', 'png' , 'webp'])

if uploaded_file is not None:
    # Read the image using PIL (required for the cropper)
    pil_image = Image.open(uploaded_file)

    # --- NEW: THE MAGNIFYING GLASS LOGIC ---
    if enable_crop:
        st.info("✂️ Draw a box around the tiny tanks below. The AI will analyze the zoomed-in region!")
        # This creates the interactive cropping tool on the screen
        cropped_pil = st_cropper(pil_image, realtime_update=True, box_color='#B80F0A')
        # Convert the cropped image back to OpenCV format for the AI
        image = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    else:
        # If crop is off, just use the whole image
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Base64 Encoding for API


    # Base64 Encoding for API
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # --- 4. CLOUD INFERENCE ---
    # The spinner will show up, run the indented code, and instantly disappear when done!
    with st.info("🔍 Running YOLO Interface... Please wait."):
        url = f"https://detect.roboflow.com/{PROJECT_ID}?api_key={API_KEY}&confidence={confidence_threshold}"
        response = requests.post(url, data=img_base64, headers={"Content-Type": "application/x-www-form-urlencoded"})

    if response.status_code == 200:
        results = response.json()
        predictions = results.get("predictions", [])
        tank_data = []

        # --- 5. GEOMETRY & MATH PIPELINE ---
        for i, pred in enumerate(predictions):
            x = int(pred['x'])
            y = int(pred['y'])
            rx = int(pred['width'] / 2)
            ry = int(pred['height'] / 2)

            cv2.ellipse(image, center=(x, y), axes=(rx, ry), angle=0, startAngle=0, endAngle=360, color=(0, 2, 255),
                        thickness=2)

            area_sqm = (math.pi * rx * ry) * (gsd_m ** 2)
            volume_m3 = area_sqm * tank_height_m

            tank_data.append({
                "Tank ID": f"Tank_{i + 1}",
                "Confidence": f"{pred['confidence'] * 100:.1f}%",
                "Footprint (Sq Meters)": round(area_sqm, 2),
                "Capacity (Cubic Meters)": round(volume_m3, 2)
            })

        # --- 6. UI DISPLAY ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(image_rgb, use_container_width=True)
            st.success(f"🎯 Total Infrastructure Count: {len(predictions)} Tanks Detected")

        with col2:
            st.subheader("Intelligence Report")
            if len(tank_data) > 0:
                df = pd.DataFrame(tank_data)
                st.metric(label="Total Facility Capacity (m³)", value=f"{df['Capacity (Cubic Meters)'].sum():,.2f}")
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Download CSV Report", data=csv, file_name='infrastructure_report.csv',
                                   mime='text/csv')
            else:
                st.warning("No infrastructure detected. Try adjusting the crop or lowering the confidence slider.")
    else:
        st.error(f"❌ API Error: {response.text}")