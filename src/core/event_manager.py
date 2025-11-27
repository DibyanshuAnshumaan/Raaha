# src/core/event_manager.py
from typing import List, Dict

class EventManager:
    def __init__(self, proximity_threshold=1.0):
        # proximity_threshold in meters — placeholder until depth sensor exists
        self.proximity_threshold = proximity_threshold

    def fuse(self, detections: List[Dict], faces: List[Dict], poses: List[Dict]) -> List[Dict]:
        events = []
        # Example rule: if person detected in center -> event "person_ahead"
        for det in detections:
            if det.get("class_name") == "person":
                x1,y1,x2,y2 = det["bbox"]
                cx = (x1 + x2) / 2
                # simple heuristic: center third of frame => ahead
                events.append({"type":"person_detected", "bbox":det["bbox"], "confidence":det.get("score")})
        # faces found -> people nearby
        if faces:
            events.append({"type":"face_detected", "count":len(faces)})
        # placeholder: poses ignored for now
        return events
