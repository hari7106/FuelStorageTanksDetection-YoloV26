import cv2
import requests
import numpy as np
import base64

# --- 1. YOUR SETTINGS ---
API_KEY = "6t3RCD7zG6anysk3AjGv"
PROJECT_ID = "mini-zwohz/1"
IMAGE_PATH = "done.jpg"

print("🚀 Translating image...")

# 2. Ping your Model (with confidence=10)
url = f"https://detect.roboflow.com/{PROJECT_ID}?api_key={API_KEY}&confidence=10"

try:
    # THE FIX: Convert the image to Base64 text format before sending!
    with open(IMAGE_PATH, "rb") as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

    response = requests.post(
        url,
        data=image_base64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    results = response.json()

    # Let's see the successful response!
    print("☁️ Raw Cloud Response:", results)

except FileNotFoundError:
    print(f"❌ Error: Could not find the image '{IMAGE_PATH}'.")
    exit()

# 3. Read the image and draw
image = cv2.imread(IMAGE_PATH)
predictions = results.get("predictions", [])

if predictions:
    print(f"✅ Found {len(predictions)} tanks! Drawing bounding circles...")

    for pred in predictions:
        cx = int(pred["x"])
        cy = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        # Calculate Radii
        a = int(w / 2)
        b = int(h / 2)

        # Draw the Bounding Circle (Ellipse)
        cv2.ellipse(image, center=(cx, cy), axes=(a, b), angle=0, startAngle=0, endAngle=360, color=(0, 0 , 255),
                    thickness=2)

        # Draw the Confidence % Text
        conf = int(pred["confidence"] * 100)

    # Show it on the screen!

    cv2.namedWindow("Tank Detector v3", cv2.WINDOW_NORMAL)
    cv2.imshow("Tank Detector v3", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Still no tanks found. Look at the Raw Cloud Response above to see why!")