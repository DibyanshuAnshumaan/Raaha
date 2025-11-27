# Raaha — Alpha Vision System for Blind Navigation

Raaha is an early-stage computer-vision framework designed to assist with navigation for people who are blind or visually impaired. The alpha version focuses exclusively on camera-based perception, providing three foundational capabilities:
  1. Object Detection
     
  2. Face Detection
     
  3. Human Pose / Skeleton Detection
   
This stage establishes a real-time perception pipeline that will later be extended with ultrasonic sensing, vibration-based haptics, audio prompts, and full sensor fusion. Raaha aims to eventually evolve into a portable wearable system capable of offering intuitive, reliable, and low-latency situational awareness.

# 1. Project Objectives
The long-term mission for Raaha is to build an assistive navigation system that:

  1. Detects and classifies obstacles

  2. Identifies humans and directional cues

  3. Analyzes body movement, posture, and path prediction

  4. Communicates essential information through haptics and audio

  5. Performs all processing locally for safety, privacy, and speed

The alpha version focuses on perception accuracy, architecture design, and establishing a modular, extensible codebase.

# 2. Features (Alpha Release)
**Object Detection**

    1. Built using YOLOv8 Nano or SSD MobileNet V2

    2. Real-time detection of common obstacles: people, vehicles, furniture, doorways, stairs, etc.

    3. Lightweight enough to run on standard CPUs

**Face Detection**

    1. MediaPipe Face Detection or OpenCV DNN

    2. Fast CPU-friendly detector

    3. Outputs bounding boxes + key landmarks

**Pose / Skeleton Detection**

    1.MediaPipe Pose or YOLOv8 Pose

    2. Extracts 33+ human keypoints

    3. Useful for understanding direction, gestures, and movement intent

**Unified Vision Pipeline**

    1. All three detectors operate on a single camera stream

    2. Event manager consolidates results for logging or further decision-making

    3. Optional visualization with bounding boxes and skeleton lines

# 3. Technology Sack
Language: Python 3.10+

CV/ML Libraries: OpenCV, MediaPipe, Ultralytics YOLO, NumPy

Model Formats: ONNX, TFLite, standard YOLO

Inference Backends: CPU, with optional ONNX Runtime / TFLite acceleration

Platform: Laptop/desktop; future: Raspberry Pi + Coral TPU

# 4. License
Copyright 2025 Dibyanshu Anshumaan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.