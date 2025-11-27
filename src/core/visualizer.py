# src/core/visualizer.py
import cv2

def draw_boxes(frame, detections, color=(0,255,0), label=True):
    for det in detections:
        x1,y1,x2,y2 = det["bbox"]
        score = det.get("score", None)
        name = det.get("class_name", "")
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        if label:
            text = f"{name}" + (f" {score:.2f}" if score is not None else "")
            cv2.putText(frame, text, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def draw_faces(frame, faces, color=(255,0,0)):
    for f in faces:
        x1,y1,x2,y2 = f["bbox"]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, "face", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

def draw_skeleton(frame, poses, color=(0,255,255)):
    for p in poses:
        keypoints = p["keypoints"]
        for (x, y, c) in keypoints:
            if c > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
