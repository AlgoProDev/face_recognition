import numpy as np
import os


class DlibResNet:
    def __init__(self):
        import dlib

        self.layers = [DlibMetaData()]

        file_name = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models",
            "lib_face_recognition_resnet_model_v1.dat",
        )

        model = dlib.face_recognition_model_v1(file_name)
        self.__model = model

    def predict(self, img_aligned):
        if len(img_aligned.shape) == 4:
            img_aligned = img_aligned[0]

        img_aligned = img_aligned[:, :, ::-1]  # bgr to rgb

        if img_aligned.max() <= 1:
            img_aligned = img_aligned * 255

        img_aligned = img_aligned.astype(np.uint8)

        model = self.__model

        img_representation = model.compute_face_descriptor(img_aligned)

        img_representation = np.array(img_representation)
        img_representation = np.expand_dims(img_representation, axis=0)

        return img_representation


class DlibMetaData:
    def __init__(self):
        self.input_shape = [[1, 150, 150, 3]]
