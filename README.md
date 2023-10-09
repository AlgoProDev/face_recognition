
# Face Recognition and Analysis
Face recognition and analysis has never been easier. "face_recognition" is a Python wrapper for deep learning-based face recognition models from famous frameworks inside a single library. With this script, you can utilize multiple face recognition models, perform verification, and even stream video to recognize faces in real-time.

## Features
- **Multiple Models Support**: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace, and SFace.
- **Face Verification**: Check if two faces are identical or belong to the same person.
- **Face Representation**: Extract numerical vector representations for a given facial image.
- **Real-time Video Stream Analysis**: Recognize and analyze faces in real-time using your webcam or other video sources.
- **Face Detection and Alignment**: Built-in support for various face detectors including opencv, retinaface, mtcnn, ssd, dlib, mediapipe, and yolov8.
- **Database Integration**: Search and identify faces within a provided database.
- **Built-in Distance Metrics**: cosine, euclidean, euclidean_l2 metrics for facial similarity checks.
- **Extendable**: Easy integration of new models and tools within the existing framework.
  
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

# Real-time Face Recognition with Video Streaming
One of the standout features of this script is the ability to perform face recognition in real-time using your webcam or any other video source. This can be used in various applications like real-time attendance systems, security systems, or even fun interactive projects.

## How to Use Real-time Streaming
1. Set Up the Facial Database:
Before streaming, you need to set up a database of facial images that the system will recognize in real-time. This database is just a directory containing images of the individuals you want to recognize. It's advised to use clear, front-facing images. Name each image file with the name of the person for easier identification later.

**Example:**

```
facial_db/
├── Olti/
    ├── olti1.png
    └── olti2.jpg
├── Gent/
    ├── gent1.png
    └── gent2.png
└── Alice/
    ├── alice1.jpg
    └── alice2.png
```
2. Start the Video Stream:
After setting up your database, you can start the real-time video stream for face recognition. Use the stream function and point it to your database:

```
stream(db_path="facial_db", model_name="VGG-Face", source=0)
```
 - **'db_path'**: Path to your facial database.
 - **'model_name'**: The face recognition model you wish to use.
 - **'source'**: Source of video stream. Set to 0 to use the default webcam. If you have multiple cameras or want to use a video file, adjust this parameter accordingly.
   
3. Adjusting Recognition Sensitivity:
If you find that the recognition is too sensitive or not sensitive enough, you can adjust the time_threshold and frame_threshold parameters.

- **'time_threshold'**: Duration (in seconds) a face is displayed after being recognized.
- **'frame_threshold'**: Number of frames required to focus on a face for it to be recognized.

```
stream(db_path="facial_db", model_name="VGG-Face", source=0, time_threshold=7, frame_threshold=10)
```
- **Notes**:
Ensure proper lighting conditions for optimal recognition.
The real-time recognition speed can vary depending on your computer's capabilities and the selected face recognition model. Some models are more computationally intensive than others.

## Setting Up the Database
The facial database is the cornerstone of any face recognition system. It's where all reference images are stored, and it's against these images that incoming faces are compared.

1. Creating the Database:
The database is just a simple directory filled with images. Each image should ideally represent one individual.

2. Image Naming:
While not compulsory, it's a good idea to name each image file with the name of the individual in the picture. This makes it easier to identify recognized faces later.

3. Image Quality:
For best results, use high-quality, clear, and front-facing images. Avoid pictures with strong shadows, with faces at an angle, or with obstructions.

4. Multiple Images per Individual:
You can include multiple images for each individual to increase the accuracy of recognition. However, ensure that each image is distinct, showing the individual in different angles, lighting conditions, or expressions.

5. Database Updates:
If you ever add or remove images from your database, ensure to remove the **```representations_{model_name}.pkl```** file in the database directory. This file caches the facial representations for faster operations, and it needs to be regenerated if the database changes.

With your database set up, you're now ready to use it for various face recognition tasks, be it real-time streaming or batch recognition. Remember always to respect privacy when collecting and storing facial images.

## Final Thoughts
Face recognition and analysis is a rapidly evolving field, with applications ranging from security to entertainment. This script offers a robust and flexible solution for both beginners and professionals looking to integrate facial recognition into their applications.

Remember, while technology is powerful, always consider privacy implications when deploying face recognition systems.
