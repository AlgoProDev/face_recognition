import numpy as np
import cv2 as cv


class _Layer:
    input_shape = (None, 112, 112, 3)
    output_shape = (None, 1, 128)


class SFaceModel:
    def __init__(self, model_path):
        self.model = cv.FaceRecognizerSF.create(
            model=model_path, config="", backend_id=0, target_id=0
        )

        self.layers = [_Layer()]

    def predict(self, image):
        input_blob = (image[0] * 255).astype(np.uint8)

        embeddings = self.model.feature(input_blob)

        return embeddings


def load_model():
    file_name = "./models/face_recognition_sface_2021dec.onnx"

    model = SFaceModel(model_path=file_name)

    return model
