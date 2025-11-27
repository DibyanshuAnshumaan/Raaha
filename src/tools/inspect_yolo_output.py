import onnxruntime as ort
import numpy as np
import os

model = r"E:\Raaha\models\yolov8n.onnx"
print("Loading model:", model)

if not os.path.exists(model):
    print("ERROR: MODEL NOT FOUND")
    exit()

sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

print("\nINPUTS:")
for i in sess.get_inputs():
    print(i.name, i.shape, i.type)

print("\nOUTPUTS:")
for o in sess.get_outputs():
    print(o.name, o.shape, o.type)

dummy = np.random.randn(1,3,640,640).astype(np.float32)
res = sess.run(None, {sess.get_inputs()[0].name: dummy})

print("\nRESULT SHAPES:")
for i, r in enumerate(res):
    print(f"[{i}] {r.shape} min={r.min()} max={r.max()} dtype={r.dtype}")
