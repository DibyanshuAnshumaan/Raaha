import cv2

class YuNetFaceDetector:
    def __init__(self, model_path, input_size=(320, 320), conf_threshold=0.8):
        self.input_w, self.input_h = input_size
        self.model = cv2.FaceDetectorYN_create(
            model=model_path,
            config="",
            input_size=(self.input_w, self.input_h),
            score_threshold=conf_threshold,
            nms_threshold=0.3,
            top_k=5000
        )

    def predict(self, frame):
        h, w = frame.shape[:2]

        # Resize for YuNet (prevents memory corruption)
        resized = cv2.resize(frame, (self.input_w, self.input_h))

        _, faces = self.model.detect(resized)
        results = []

        if faces is not None:
            for face in faces:
                x, y, fw, fh = face[:4]

                # Scale back to original image
                x = int(x * w / self.input_w)
                y = int(y * h / self.input_h)
                fw = int(fw * w / self.input_w)
                fh = int(fh * h / self.input_h)

                results.append({
                    "bbox": [x, y, x + fw, y + fh],
                    "score": float(face[14]) if len(face) > 14 else 1.0
                })

        return results
