# tools/verify_yolo_onnx.py
import onnxruntime as ort
import numpy as np
import os
model = os.path.join("models","yolov8s.onnx")
sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
outs = sess.get_outputs()
print("INPUT:", inp.name, inp.shape, inp.type)
for o in outs:
    print("OUTPUT:", o.name, o.shape, o.type)
# forward pass
dummy = np.random.randn(1,3,640,640).astype(np.float32)
out = sess.run(None, {inp.name: dummy})
print("Forward outputs shapes:", [o.shape for o in out])
