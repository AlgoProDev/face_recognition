from . import Facenet
import os


def loadModel():
    model = Facenet.InceptionResNetV2(dimension=512)

    file_name = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "facenet512_weights.h5",
    )

    model.load_weights(file_name)

    return model
