# src/camera_stream.py
import cv2

class CameraStream:
    def __init__(self, src=0, width=1280, height=720, backend=cv2.CAP_DSHOW):
        self.src = src

        # Use DirectShow (more stable than MSMF)
        self.cap = cv2.VideoCapture(src, backend)

        # Apply resolution safely
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Disable autofocus flicker if supported
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()
