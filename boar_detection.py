# boar_detection.py
# Real-time boar detection using YOLOv8 and GPIO control for deterrents

import cv2
import numpy as np
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPIO setup for actuators (lights and sound)
LIGHT_PIN = 17   # GPIO pin for LED/strobe light
SOUND_PIN = 27   # GPIO pin for sound module
GPIO.setmode(GPIO.BCM)
GPIO.setup(LIGHT_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(SOUND_PIN, GPIO.OUT, initial=GPIO.LOW)

# Load trained YOLOv8 model (replace with your model path)
model = YOLO('best_boar.pt')  # Path to your trained weights

# Detection parameters
CONFIDENCE_THRESHOLD = 0.6   # Minimum confidence for detection
DETECTION_COOLDOWN = 10       # Seconds to wait before triggering again
last_trigger_time = 0

# Camera setup (use 0 for default USB camera, or video file for testing)
cap = cv2.VideoCapture(0)  # For Raspberry Pi Camera: use 'nvarguscamerasrc ! ...' pipeline

def trigger_deterrent():
    """Activate lights and sound to scare away boars"""
    logger.info("Boar detected! Triggering deterrents...")
    GPIO.output(LIGHT_PIN, GPIO.HIGH)
    GPIO.output(SOUND_PIN, GPIO.HIGH)
    time.sleep(5)  # Keep on for 5 seconds
    GPIO.output(LIGHT_PIN, GPIO.LOW)
    GPIO.output(SOUND_PIN, GPIO.LOW)

def detect_boars(frame):
    """Run YOLO inference and return detected boar bounding boxes"""
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # Assuming class 0 is 'boar' (you can adjust based on your dataset)
                if cls == 0 and conf >= CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, conf))
    return detections

def main():
    global last_trigger_time
    logger.info("Starting AgriGuard boar detection system...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame")
            break
        
        # Run detection
        boars = detect_boars(frame)
        
        # Annotate frame for display
        for (x1, y1, x2, y2, conf) in boars:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Boar: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Trigger deterrent if any boar detected and cooldown elapsed
        now = time.time()
        if boars and (now - last_trigger_time > DETECTION_COOLDOWN):
            trigger_deterrent()
            last_trigger_time = now
        
        # Show output (optional, can be omitted for headless deployment)
        cv2.imshow("AgriGuard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

if __name__ == "__main__":
    main()
