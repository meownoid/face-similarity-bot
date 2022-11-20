import os

TOKEN = "<TOKEN>"
DOWNLOAD_ASSETS = True
PHOTOS_DIR = "photos"
ASSETS_DIR = "assets"
LOG_LEVEL = "DEBUG"

LANDMARKS_URL = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
LANDMARKS_PATH = os.path.join(ASSETS_DIR, "shape_predictor_5_face_landmarks.dat")

MODEL_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
MODEL_PATH = os.path.join(ASSETS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
