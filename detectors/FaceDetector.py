import math
from PIL import Image
import numpy as np
from ..commons import distance
from . import OpenCvWrapper


def build_model(detector_backend):
    global face_detector_obj

    backends = {
        "opencv": OpenCvWrapper.build_model,
    }

    if not "face_detector_obj" in globals():
        face_detector_obj = {}

    built_models = list(face_detector_obj.keys())
    if detector_backend not in built_models:
        face_detector = backends.get(detector_backend)

        if face_detector:
            face_detector = face_detector()
            face_detector_obj[detector_backend] = face_detector
        else:
            raise ValueError("invalid detector_backend passed - " + detector_backend)

    return face_detector_obj[detector_backend]


def detect_faces(face_detector, detector_backend, img, align=True):
    backends = {
        "opencv": OpenCvWrapper.detect_face,
    }

    detect_face_fn = backends.get(detector_backend)

    if detect_face_fn:
        obj = detect_face_fn(face_detector, img, align)
        return obj
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)


def alignment_procedure(img, left_eye, right_eye):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1

    a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    if b != 0 and c != 0:
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    return img
