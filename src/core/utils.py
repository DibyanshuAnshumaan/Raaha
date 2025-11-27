# src/utils.py
import time

def fps_timer():
    t0 = time.time()
    while True:
        t1 = time.time()
        dt = t1 - t0
        fps = 1.0 / dt if dt > 0 else 0
        t0 = t1
        yield fps
