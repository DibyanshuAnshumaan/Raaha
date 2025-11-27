import onnxruntime as ort
import numpy as np
import cv2

class YOLOv8PoseONNX:
    def __init__(self, model_path, input_size=(640, 640), conf_thresh=0.25):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_thresh = conf_thresh

    def preprocess(self, image):
        img = cv2.resize(image, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, frame):
        img_input = self.preprocess(frame)
        output = self.session.run(None, {self.input_name: img_input})[0]  # shape: (1, N, 56)
        output = output[0]  # remove batch dimension

        poses = []
        img_h, img_w = frame.shape[:2]

        for row in output:
            conf = float(row[4])  # extract object confidence as scalar
            if conf < self.conf_thresh:
                continue
            
            # YOLOv8 format: x_center, y_center, w, h
            x_center, y_center, w, h = row[0], row[1], row[2], row[3]

            x1 = int((x_center - w/2) * img_w / self.input_size[0])
            y1 = int((y_center - h/2) * img_h / self.input_size[1])
            x2 = int((x_center + w/2) * img_w / self.input_size[0])
            y2 = int((y_center + h/2) * img_h / self.input_size[1])

            keypoints = []
            kp_data = row[6:]  # skip: x,y,w,h,conf,cls

            for i in range(0, len(kp_data), 3):
                kx = int(kp_data[i])     # no scaling
                ky = int(kp_data[i+1])   # no scaling

                kc = float(kp_data[i+2])
                keypoints.append([kx, ky, kc])

            poses.append({
                "bbox": [x1, y1, x2, y2],
                "keypoints": keypoints,
                "score": conf
            })

        return poses
