import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from face_recognition.basemodels.DlibResNet import DlibResNet


def loadModel():
    return DlibResNet()
