import cv2
import os

from camera_stream import CameraStream
from detectors.object_detector_onnx import YOLOv8ONNX
from detectors.face_detector_yunet import YuNetFaceDetector
from core.visualizer import draw_boxes, draw_faces
from core.utils import fps_timer


def main():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    # Use YOLOv8-NANO
    yolov8_path = os.path.abspath(os.path.join(models_dir, "yolov8n.onnx"))
    yunet_path  = os.path.abspath(os.path.join(models_dir, "yunet.onnx"))

    if not os.path.exists(yolov8_path):
        print("ERROR: Missing YOLO model:", yolov8_path)
        return

    if not os.path.exists(yunet_path):
        print("ERROR: Missing YuNet model:", yunet_path)
        return

    # Camera
    cam = CameraStream(src=0, width=640, height=480)

    # Detectors
    detector = YOLOv8ONNX(yolov8_path, conf=0.60, iou=0.45)
    face_detector = YuNetFaceDetector(yunet_path)

    fps_gen = fps_timer()

    frame_id = 0
    last_detections = []

    while True:
        frame = cam.read()
        if frame is None:
            print("No frame received. Exiting.")
            break

        frame_id += 1

        # YOLO every 3 frames
        if frame_id % 3 == 0:
            last_detections = detector.predict(frame)

        detections = last_detections
        faces = face_detector.predict(frame)

        # Draw results
        draw_boxes(frame, detections)
        draw_faces(frame, faces)

        # FPS
        fps = next(fps_gen)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        cv2.imshow("Raaha Alpha", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
