
# Face Recognition and Analysis
Face recognition and analysis has never been easier. "face_recognition" is a Python wrapper for deep learning-based face recognition models from famous frameworks inside a single library. With this script, you can utilize multiple face recognition models, perform verification, and even stream video to recognize faces in real-time.

## Features
- Multiple Models Support: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace, and SFace.
- Face Verification: Check if two faces are identical or belong to the same person.
- Face Representation: Extract numerical vector representations for a given facial image.
- Real-time Video Stream Analysis: Recognize and analyze faces in real-time using your webcam or other video sources.
- Face Detection and Alignment: Built-in support for various face detectors including opencv, retinaface, mtcnn, ssd, dlib, mediapipe, and yolov8.
- Database Integration: Search and identify faces within a provided database.
- Built-in Distance Metrics: cosine, euclidean, euclidean_l2 metrics for facial similarity checks.
- Extendable: Easy integration of new models and tools within the existing framework.
## How to Use
1. Build a FaceComparer Model:
```
model = build_model("VGG-Face")
```
2. Face Verification:
```
response = verify("img1.jpg", "img2.jpg", model_name="VGG-Face", distance_metric="cosine")
```
3. Face Representation:
```
embeddings = represent("img.jpg", model_name="VGG-Face")
```
4. Real-time Streaming:
```
stream(db_path="facial_db", model_name="VGG-Face", source=0)
```
5. Search in Database:
```
results = find("target_img.jpg", db_path="facial_db", model_name="VGG-Face")
```
## Requirements
- Python 3.6+
- Tensorflow 2.x
- OpenCV
- Numpy
- Pandas
- TQDM
- Dlib
## Installation
Before you begin, ensure you have the required libraries installed. You can generally install them with pip:

```
pip install -r requirements.txt
```

## Final Thoughts
Face recognition and analysis is a rapidly evolving field, with applications ranging from security to entertainment. This script offers a robust and flexible solution for both beginners and professionals looking to integrate facial recognition into their applications.

Remember, while technology is powerful, always consider privacy implications when deploying face recognition systems.
