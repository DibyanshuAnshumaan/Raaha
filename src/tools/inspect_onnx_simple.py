import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("models/yolov8s.onnx")
inp = sess.get_inputs()[0]
print("INPUT:", inp.shape)

out = sess.get_outputs()[0]
print("OUTPUT:", out.shape)

print("\nFirst 10 values of output:")
pred = sess.run(None, {inp.name: np.random.randn(1,3,640,640).astype(np.float32)})[0]
print(pred.flatten()[:50])
