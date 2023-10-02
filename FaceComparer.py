import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


import warnings
import time
import logging
import numpy as np
import cv2
import tensorflow as tf
from face_recognition.basemodels import (
    VGGFace,
    OpenFace,
    Facenet,
    Facenet512,
    FbDeepFace,
    DlibWrapper,
    ArcFace,
    SFace,
)
from face_recognition.commons import functions, distance as dst

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel(logging.ERROR)

available_models = [
    "VGG-Face",
    "OpenFace",
    "Facenet",
    "Facenet512",
    "DeepFace",
    "Dlib",
    "ArcFace",
    "SFace",
]


def build_model(model_name):
    """
    This function builds a deepface model
    Parameters:
            model_name (string): face recognition or facial attribute model
                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
                    Age, Gender, Emotion, Race for facial attributes

    Returns:
            built deepface model
    """

    global model_obj

    models = {
        "VGG-Face": VGGFace.loadModel,
        "OpenFace": OpenFace.loadModel,
        "Facenet": Facenet.loadModel,
        "Facenet512": Facenet512.loadModel,
        "DeepFace": FbDeepFace.loadModel,
        "Dlib": DlibWrapper.loadModel,
        "ArcFace": ArcFace.loadModel,
        "SFace": SFace.load_model,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj:
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")

    return model_obj[model_name]


def verify(
    img1_path,
    img2_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enforce_detection=True,
    align=True,
    normalization="base",
):
    """
    This function verifies an image pair is same person or different persons. In the background,
    verification function represents facial images as vectors and then calculates the similarity
    between those vectors. Vectors of same person images should have more similarity (or less
    distance) than vectors of different persons.

    Parameters:
            img1_path, img2_path: exact image path as string. numpy array (BGR) or based64 encoded
            images are also welcome. If one of pair has more than one face, then we will compare the
            face pair with max similarity.

            model_name (str): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib
            , ArcFace and SFace

            distance_metric (string): cosine, euclidean, euclidean_l2

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib, mediapipe or yolov8.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Verify function returns a dictionary.

            {
                    "verified": True
                    , "distance": 0.2563
                    , "max_threshold_to_verify": 0.40
                    , "model": "VGG-Face"
                    , "similarity_metric": "cosine"
                    , 'facial_areas': {
                            'img1': {'x': 345, 'y': 211, 'w': 769, 'h': 769},
                            'img2': {'x': 318, 'y': 534, 'w': 779, 'h': 779}
                    }
                    , "time": 2
            }

    """

    tic = time.time()

    target_size = functions.find_target_size(model_name=model_name)

    img1_objs = functions.extract_faces(
        img=img1_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    img2_objs = functions.extract_faces(
        img=img2_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )
    distances = []
    regions = []
    for img1_content, img1_region, _ in img1_objs:
        for img2_content, img2_region, _ in img2_objs:
            img1_embedding_obj = represent(
                img_path=img1_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img2_embedding_obj = represent(
                img_path=img2_content,
                model_name=model_name,
                enforce_detection=enforce_detection,
                detector_backend="skip",
                align=align,
                normalization=normalization,
            )

            img1_representation = img1_embedding_obj[0]["embedding"]
            img2_representation = img2_embedding_obj[0]["embedding"]

            if distance_metric == "cosine":
                distance = dst.findCosineDistance(
                    img1_representation, img2_representation
                )
            elif distance_metric == "euclidean":
                distance = dst.findEuclideanDistance(
                    img1_representation, img2_representation
                )
            elif distance_metric == "euclidean_l2":
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(img1_representation),
                    dst.l2_normalize(img2_representation),
                )
            else:
                raise ValueError("Invalid distance_metric passed - ", distance_metric)

            distances.append(distance)
            regions.append((img1_region, img2_region))

    threshold = dst.findThreshold(model_name, distance_metric)
    distance = min(distances)
    facial_areas = regions[np.argmin(distances)]

    toc = time.time()

    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "detector_backend": detector_backend,
        "similarity_metric": distance_metric,
        "facial_areas": {"img1": facial_areas[0], "img2": facial_areas[1]},
        "time": round(toc - tic, 2),
    }

    return resp_obj


def represent(
    img_path,
    model_name="VGG-Face",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
):
    """
    This function represents facial images as vectors. The function uses convolutional neural
    networks models to generate vector embeddings.

    Parameters:
            img_path (string): exact image path. Alternatively, numpy array (BGR) or based64
            encoded images could be passed. Source image can have many faces. Then, result will
            be the size of number of faces appearing in the source image.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
            ArcFace, SFace

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib, mediapipe or yolov8.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Represent function returns a list of object with multidimensional vector (embedding).
            The number of dimensions is changing based on the reference model.
            E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
    """
    resp_objs = []

    model = build_model(model_name)

    target_size = functions.find_target_size(model_name=model_name)
    if detector_backend != "skip":
        img_objs = functions.extract_faces(
            img=img_path,
            target_size=target_size,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
        )
    else:
        if isinstance(img_path, str):
            img = functions.load_image(img_path)
        elif type(img_path).__module__ == np.__name__:
            img = img_path.copy()
        else:
            raise ValueError(f"unexpected type for img_path - {type(img_path)}")
        if len(img.shape) == 4:
            img = img[0]
        if len(img.shape) == 3:
            img = cv2.resize(img, target_size)
            img = np.expand_dims(img, axis=0)
        img_region = [0, 0, img.shape[1], img.shape[0]]
        img_objs = [(img, img_region, 0)]

    for img, region, confidence in img_objs:
        img = functions.normalize_input(img=img, normalization=normalization)

        if "keras" in str(type(model)):
            embedding = model.predict(img, verbose=0)[0].tolist()
        else:
            embedding = model.predict(img)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs
