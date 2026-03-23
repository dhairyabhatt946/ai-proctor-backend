import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import io
from architecture import ProctoringCNN

device = torch.device('cpu') 
model = ProctoringCNN()
model.load_state_dict(torch.load('master_model.pth', map_location=device))
model.eval()

data_pipeline = transforms.Compose([
    transforms.Resize((64, 64)),                 
    transforms.ToTensor()                        
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def analyze_frame(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return "Violation: Face Not Visible (Turned Away or Left Room)"
        
    # Crop the face out of the room so the AI gets a high-resolution view!
    x, y, w, h = faces[0]
    face_crop = frame[y:y+h, x:x+w]
    
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    tensor = data_pipeline(face_pil).unsqueeze(0) 
    
    with torch.no_grad():
        coords = model(tensor)[0].numpy()
        
    left_corner_x_ratio = coords[0]
    right_corner_x_ratio = coords[4]
    pupil_x_ratio = coords[8]
    
    eye_width_ratio = abs(right_corner_x_ratio - left_corner_x_ratio)
    
    if eye_width_ratio < 0.01:
        return "Violation: Eye Structure Lost / Anomaly"
        
    relative_pupil_ratio = (pupil_x_ratio - left_corner_x_ratio) / eye_width_ratio
    
    print(f"Deep Learning Pupil Tracking Ratio: {relative_pupil_ratio:.2f}")
    
    if relative_pupil_ratio < 0.35 or relative_pupil_ratio > 0.65:
         return "Violation: Off-Screen Gaze Detected"

    return "Focused"