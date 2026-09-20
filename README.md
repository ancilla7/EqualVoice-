# EqualVoice AI

EqualVoice AI is an experimental computer-vision project focused on isolated sign-language recognition. The goal of the project is to explore how standard webcam feeds and machine learning can be used to detect and interpret hand gestures into readable text in real time.

> **Note on Project Scope:**  
> EqualVoice AI is currently an early-stage prototype and does **not** provide full or fluent sign language translation. Natural sign languages possess rich grammar, spatial context, continuous movement, and facial expressions. The initial version is strictly limited to recognizing a small, predefined set of isolated signs/gestures.

---

## Initial MVP

The primary objective of the Minimum Viable Product (MVP) is to establish an end-to-end working pipeline:
- Capture video input from a standard webcam.
- Detect and track hand landmarks in real time.
- Classify a limited dictionary of predefined static signs.
- Output and display the recognized sign as text.

---

## High-Level Pipeline

The planned processing flow operates in five main stages:

```
[ Webcam Input ]
       │
       ▼
[ Hand Landmark Detection ]   (Detect 21 key landmarks per hand)
       │
       ▼
[ Feature Extraction ]        (Coordinate normalization, relative distances/angles)
       │
       ▼
[ ML Classifier ]             (Trained model mapping features to predefined classes)
       │
       ▼
[ Text Output ]               (Real-time on-screen prediction)
```

1. **Webcam Input**: Captures live frames from a consumer-grade camera.
2. **Hand Landmark Detection**: Locates 2D/3D coordinates of key hand joints.
3. **Feature Extraction**: Normalizes landmark coordinates to be invariant to hand size and position within the frame.
4. **ML Classifier**: Evaluates normalized landmark features to predict the corresponding sign class.
5. **Text Output**: Displays the predicted class label to the user interface.

---

## Project Status

- **Phase**: Architecture & Design (MVP Planning)
- Application code and dependencies will be configured in subsequent milestones.
