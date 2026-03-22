# import torch
# from architecture import ProctoringCNN
# import cv2
# import numpy as np
# import torchvision.transforms as transforms
# from PIL import Image
# import io

# # 1. Load YOUR Custom AI Brain (Tier 1)
# device = torch.device('cpu') 
# model = ProctoringCNN()
# model.load_state_dict(torch.load('master_model.pth', map_location=device))
# model.eval()

# data_pipeline = transforms.Compose([
#     transforms.Resize((64, 64)),                 
#     transforms.ToTensor()                        
# ])

# # 2. Load OpenCV Face Detector ONLY (We deleted the weak eye detector!)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def analyze_frame(image_bytes):
#     # --- TIER 1: PYTORCH MACRO-VISION ---
#     image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     tensor = data_pipeline(image_pil).unsqueeze(0) 
    
#     with torch.no_grad():
#         coords = model(tensor)[0].numpy()
        
#     ai_left_corner_x = coords[0]
#     ai_right_corner_x = coords[4]
#     ai_eye_width = abs(ai_right_corner_x - ai_left_corner_x)
    
#     if ai_eye_width < 50.0:
#         return "Violation: Camera Obstructed / Too Close"

#     # --- TIER 2: OPENCV ANATOMY-VISION ---
#     nparr = np.frombuffer(image_bytes, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Check 1: Did they turn around, or leave the room entirely?
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#     if len(faces) == 0:
#         return "Violation: Face Not Visible (Turned Away or Left Room)"
        
#     x, y, w, h = faces[0]
#     face_gray = gray[y:y+h, x:x+w]
    
#     # Anatomical Cropping
#     ey1 = int(h * 0.20)  
#     ey2 = int(h * 0.50)  
#     ex1 = int(w * 0.10)  
#     ex2 = int(w * 0.45)  
    
#     eye_crop = face_gray[ey1:ey2, ex1:ex2]
    
#     _, threshold = cv2.threshold(eye_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Check 2: Are the eyes covered, closed, or completely lost?
#     if not contours:
#         return "Violation: Eyes Closed or Covered"
    
#     # Check 3: Where is the pupil looking?
#     contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
#     (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
    
#     custom_eye_width = ex2 - ex1
#     pupil_ratio = cx / custom_eye_width
    
#     print(f"PyTorch Check: Passed | Pure Pupil Ratio: {pupil_ratio:.2f}")
    
#     if pupil_ratio < 0.30 or pupil_ratio > 0.70:
#          return "Violation: Off-Screen Gaze Detected"

#     # If it survives all the checks, they are behaving!
#     return "Focused"

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import io
from architecture import ProctoringCNN

# 1. Load YOUR Retrained, Normalized AI Brain
device = torch.device('cpu') 
model = ProctoringCNN()
model.load_state_dict(torch.load('master_model.pth', map_location=device))
model.eval()

# Preprocessing Pipeline for the AI
data_pipeline = transforms.Compose([
    transforms.Resize((64, 64)),                 
    transforms.ToTensor()                        
])

# 2. Load OpenCV Face Detector (Used ONLY as a magnifying glass)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_frame(image_bytes):
    # Decode the webcam image
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # --- STEP 1: MACRO-LOCALIZATION (OpenCV Magnifying Glass) ---
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return "Violation: Face Not Visible (Turned Away or Left Room)"
        
    # Crop the face out of the room so the AI gets a high-resolution view!
    x, y, w, h = faces[0]
    face_crop = frame[y:y+h, x:x+w]
    
    # --- STEP 2: MICRO-TRACKING (Your PyTorch CNN) ---
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    tensor = data_pipeline(face_pil).unsqueeze(0) 
    
    # Run your trained weights!
    with torch.no_grad():
        coords = model(tensor)[0].numpy()
        
    # Because you normalized the training data, these are now perfect Ratios (0.0 to 1.0)
    left_corner_x_ratio = coords[0]
    right_corner_x_ratio = coords[4]
    pupil_x_ratio = coords[8]
    
    # Calculate the width of the eye in the AI's mathematical space
    eye_width_ratio = abs(right_corner_x_ratio - left_corner_x_ratio)
    
    if eye_width_ratio < 0.01:
        return "Violation: Eye Structure Lost / Anomaly"
        
    # Calculate exactly where the pupil is horizontally inside the eye
    relative_pupil_ratio = (pupil_x_ratio - left_corner_x_ratio) / eye_width_ratio
    
    print(f"Deep Learning Pupil Tracking Ratio: {relative_pupil_ratio:.2f}")
    
    # --- FINAL LOGIC THRESHOLDS ---
    # Normal looking forward is around 0.50
    if relative_pupil_ratio < 0.35 or relative_pupil_ratio > 0.65:
         return "Violation: Off-Screen Gaze Detected"

    return "Focused"