import onnxruntime as ort
import numpy as np
import cv2

COCO_CLASSES = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
 "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
 "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
 "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
 "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
 "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
 "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
 "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
 "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
 "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


class YOLOv8ONNX:
    """
    Decoder for YOLOv8 ONNX exporting normalized XYWH and class probabilities.
    Output shape: (1, 84, 8400)
    boxes: normalized xywh (0..1)
    classes: probabilities (0..1)
    """
    def __init__(self, path, conf=0.25, iou=0.45):
        self.conf_thr = conf
        self.iou_thr = iou

        so = ort.SessionOptions()
        so.log_severity_level = 3

        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def preprocess(self, img):
        h0, w0 = img.shape[:2]
        img_r = cv2.resize(img, (640, 640))
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        img_f = img_r.astype(np.float32) / 255.0
        img_f = np.transpose(img_f, (2, 0, 1))[None]
        return img_f, h0, w0

    def nms(self, boxes, scores, iou_thr):
        x1 = boxes[:, 0]; y1 = boxes[:, 1]
        x2 = boxes[:, 2]; y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, out, h0, w0):
        # out shape: (1, 84, 8400) → flatten to (8400, 84)
        out = out.reshape(84, -1).T  # (8400, 84)

        # Split
        xywh = out[:, 0:4]       # normalized [0..1]
        cls_probs = out[:, 4:]   # already probabilities

        # Best class per box
        scores = cls_probs.max(axis=1)
        cls_ids = cls_probs.argmax(axis=1)

        keep = scores > self.conf_thr
        if not np.any(keep):
            return []

        xywh = xywh[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # Convert normalized xywh → pixel xyxy
        x = xywh[:, 0] * w0
        y = xywh[:, 1] * h0
        w = xywh[:, 2] * w0
        h = xywh[:, 3] * h0

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep_idx = self.nms(boxes, scores, self.iou_thr)

        results = []
        for i in keep_idx:
            box = boxes[i].astype(int)
            cid = int(cls_ids[i])
            results.append({
                "bbox": box.tolist(),
                "score": float(scores[i]),
                "class_id": cid,
                "class_name": COCO_CLASSES[cid]
            })
        return results

    def predict(self, img):
        try:
            inp, h0, w0 = self.preprocess(img)
            out = self.sess.run(None, {self.input: inp})
            return self.postprocess(out[0], h0, w0)
        except Exception as e:
            print("YOLO error:", e)
            return []
