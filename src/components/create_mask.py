import cv2
import numpy as np
from PIL import Image

def create_mask(input_image):
    if input_image is None:
        print(f"No image provided.")
        exit()

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # --- Faces detection ---
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15)

    # --- Inverse mask creation ---
    mask = np.ones(input_image.shape[:2], np.uint8) * 255  # Maschera bianca (tutto da mantenere)
    enlargement = 0
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Create a BLACK mask (to inpaint) in the face region
            cv2.rectangle(mask, (x, y-enlargement), (x+w, y+h), 0, -1) #0 per il nero, -1 per riempire
            ## If you want, you can use an ellipse for a smoother transition
            # center_coordinates = (x + w // 2, y + h // 2)
            # axesLength = (w // 2, h // 2)
            # angle = 0
            # startAngle = 0
            # endAngle = 360
            # cv2.ellipse(mask, center_coordinates, axesLength, angle, startAngle, endAngle, 0, -1)

    else:
        print("No face detected.")
        exit()

    cv2.imwrite("inverse_mask.png", mask) 
    return mask