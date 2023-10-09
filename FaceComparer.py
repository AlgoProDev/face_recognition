import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


import warnings
import time
import pickle
import logging
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from .basemodels import (
    VGGFace,
    OpenFace,
    Facenet,
    Facenet512,
    FbDeepFace,
    DlibWrapper,
    ArcFace,
    SFace,
)
from .commons import functions, distance as dst, realtime

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
            model_name (string): face recognition
                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition

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

            model_name (str): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, Dlib
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


def extract_faces(
    img_path,
    target_size=(224, 224),
    detector_backend="opencv",
    enforce_detection=True,
    align=True,
    grayscale=False,
):
    """
    This function applies pre-processing stages of a face recognition pipeline
    including detection and alignment

    Parameters:
            img_path: exact image path, numpy array (BGR) or base64 encoded image.
            Source image can have many face. Then, result will be the size of number
            of faces appearing in that source image.

            target_size (tuple): final shape of facial image. black pixels will be
            added to resize the image.

            detector_backend (string): face detection backends are retinaface, mtcnn,
            opencv, ssd or dlib

            enforce_detection (boolean): function throws exception if face cannot be
            detected in the fed image. Set this to False if you do not want to get
            an exception and run the function anyway.

            align (boolean): alignment according to the eye positions.

            grayscale (boolean): extracting faces in rgb or gray scale

    Returns:
            list of dictionaries. Each dictionary will have facial image itself,
            extracted area from the original image and confidence score.

    """

    resp_objs = []
    img_objs = functions.extract_faces(
        img=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=grayscale,
        enforce_detection=enforce_detection,
        align=align,
    )

    for img, region, confidence in img_objs:
        resp_obj = {}

        # discard expanded dimension
        if len(img.shape) == 4:
            img = img[0]

        resp_obj["face"] = img[:, :, ::-1]
        resp_obj["facial_area"] = region
        resp_obj["confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs


def find(
    img_path,
    db_path,
    model_name="VGG-Face",
    distance_metric="cosine",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    normalization="base",
    silent=False,
):
    """
    This function applies verification several times and find the identities in a database

    Parameters:
            img_path: exact image path, numpy array (BGR) or based64 encoded image.
            Source image can have many faces. Then, result will be the size of number of
            faces in the source image.

            db_path (string): You should store some image files in a folder and pass the
            exact folder path to this. A database image can also have many faces.
            Then, all detected faces in db side will be considered in the decision.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID,
            Dlib, ArcFace, SFace or Ensemble

            distance_metric (string): cosine, euclidean, euclidean_l2

            enforce_detection (bool): The function throws exception if a face could not be detected.
            Set this to False if you don't want to get exception. This might be convenient for low
            resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib, mediapipe or yolov8.

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

            silent (boolean): disable some logging and progress bars

    Returns:
            This function returns list of pandas data frame. Each item of the list corresponding to
            an identity in the img_path.
    """

    tic = time.time()

    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    target_size = functions.find_target_size(model_name=model_name)

    # ---------------------------------------

    file_name = f"representations_{model_name}.pkl"
    file_name = file_name.replace("-", "_").lower()

    if os.path.exists(db_path + "/" + file_name):
        if not silent:
            print(
                f"WARNING: Representations for images in {db_path} folder were previously stored"
                + f" in {file_name}. If you added new instances after the creation, then please "
                + "delete this file and call find function again. It will create it again."
            )

        with open(f"{db_path}/{file_name}", "rb") as f:
            representations = pickle.load(f)

        if not silent:
            print(
                "There are ",
                len(representations),
                " representations found in ",
                file_name,
            )

    else:  # create representation.pkl from scratch
        employees = []

        for r, _, f in os.walk(db_path):
            for file in f:
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    exact_path = r + "/" + file
                    employees.append(exact_path)

        if len(employees) == 0:
            raise ValueError(
                "There is no image in ",
                db_path,
                " folder! Validate .jpg or .png files exist in this path.",
            )

        # ------------------------
        # find representations for db images

        representations = []

        # for employee in employees:
        pbar = tqdm(
            range(0, len(employees)),
            desc="Finding representations",
            disable=silent,
        )
        for index in pbar:
            employee = employees[index]

            img_objs = functions.extract_faces(
                img=employee,
                target_size=target_size,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
            )

            for img_content, _, _ in img_objs:
                embedding_obj = represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]

                instance = []
                instance.append(employee)
                instance.append(img_representation)
                representations.append(instance)

        # -------------------------------

        with open(f"{db_path}/{file_name}", "wb") as f:
            pickle.dump(representations, f)

        if not silent:
            print(
                f"Representations stored in {db_path}/{file_name} file."
                + "Please delete this file when you add new identities in your database."
            )

    # ----------------------------
    # now, we got representations for facial database
    df = pd.DataFrame(
        representations, columns=["identity", f"{model_name}_representation"]
    )

    # img path might have more than once face
    target_objs = functions.extract_faces(
        img=img_path,
        target_size=target_size,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    resp_obj = []

    for target_img, target_region, _ in target_objs:
        target_embedding_obj = represent(
            img_path=target_img,
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )

        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = target_region["x"]
        result_df["source_y"] = target_region["y"]
        result_df["source_w"] = target_region["w"]
        result_df["source_h"] = target_region["h"]

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]

            if distance_metric == "cosine":
                distance = dst.findCosineDistance(
                    source_representation, target_representation
                )
            elif distance_metric == "euclidean":
                distance = dst.findEuclideanDistance(
                    source_representation, target_representation
                )
            elif distance_metric == "euclidean_l2":
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(source_representation),
                    dst.l2_normalize(target_representation),
                )
            else:
                raise ValueError(f"invalid distance metric passes - {distance_metric}")

            distances.append(distance)

            # ---------------------------

        result_df[f"{model_name}_{distance_metric}"] = distances

        threshold = dst.findThreshold(model_name, distance_metric)
        result_df = result_df.drop(columns=[f"{model_name}_representation"])
        result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
        result_df = result_df.sort_values(
            by=[f"{model_name}_{distance_metric}"], ascending=True
        ).reset_index(drop=True)

        resp_obj.append(result_df)

    # -----------------------------------

    toc = time.time()

    if not silent:
        print("find function lasts ", toc - tic, " seconds")

    return resp_obj


def stream(
    db_path="",
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=True,
    source=0,
    time_threshold=5,
    frame_threshold=5,
):
    """
    This function applies real time face recognition and facial attribute analysis

    Parameters:
            db_path (string): facial database path. You should store some .jpg files in this folder.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
            ArcFace, SFace

            detector_backend (string): opencv, retinaface, mtcnn, ssd, dlib, mediapipe or yolov8.

            distance_metric (string): cosine, euclidean, euclidean_l2

            enable_facial_analysis (boolean): Set this to False to just run face recognition

            source: Set this to 0 for access web cam. Otherwise, pass exact video path.

            time_threshold (int): how many second analyzed image will be displayed

            frame_threshold (int): how many frames required to focus on face

    """

    if time_threshold < 1:
        raise ValueError(
            "time_threshold must be greater than the value 1 but you passed "
            + str(time_threshold)
        )

    if frame_threshold < 1:
        raise ValueError(
            "frame_threshold must be greater than the value 1 but you passed "
            + str(frame_threshold)
        )

    realtime.analysis(
        db_path,
        model_name,
        detector_backend,
        distance_metric,
        source=source,
        time_threshold=time_threshold,
        frame_threshold=frame_threshold,
    )
