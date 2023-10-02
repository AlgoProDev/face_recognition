import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from face_recognition.basemodels import Facenet


def loadModel():
    model = Facenet.InceptionResNetV2(dimension=512)

    model.load_weights("./models/facenet512_weights.h5")

    return model
