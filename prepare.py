import bz2
import hashlib
import os
import pickle
import shutil
import sys
import tempfile
import urllib.request

import dlib
import numpy as np
from PIL import Image

import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def download_unpack_bz2(url, path):
    tmp_path = os.path.join(
        tempfile.gettempdir(), hashlib.md5(url.encode("UTF-8")).hexdigest()
    )
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp_path)
    print(f"Unpacking {url}")
    os.makedirs(os.path.dirname(config.LANDMARKS_PATH), exist_ok=True)
    with bz2.open(tmp_path, "rb") as zf, open(path, "wb") as ff:
        shutil.copyfileobj(zf, ff)


if not os.path.exists(config.LANDMARKS_PATH) and config.DOWNLOAD_ASSETS:
    download_unpack_bz2(config.LANDMARKS_URL, config.LANDMARKS_PATH)

if not os.path.exists(config.MODEL_PATH) and config.DOWNLOAD_ASSETS:
    download_unpack_bz2(config.MODEL_URL, config.MODEL_PATH)

if not os.path.exists(config.LANDMARKS_PATH):
    print(f"{config.LANDMARKS_PATH} does not exist")
    sys.exit(1)

if not os.path.exists(config.MODEL_PATH):
    print(f"{config.MODEL_PATH}  does not exist")
    sys.exit(1)

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(config.LANDMARKS_PATH)
face_recognition_model = dlib.face_recognition_model_v1(config.MODEL_PATH)

fs = os.listdir(config.PHOTOS_DIR)
es = []

for f in fs:
    _, ext = os.path.splitext(f)

    if ext not in IMAGE_EXTENSIONS:
        print(f"{f}: not an image, skipping")
        continue

    print(f"{f}: processing")

    image = np.asarray(Image.open(os.path.join(config.PHOTOS_DIR, f)))
    face_detects = face_detector(image, 1)
    face = face_detects[0]
    landmarks = shape_predictor(image, face)
    embedding = face_recognition_model.compute_face_descriptor(
        image, landmarks, num_jitters=10
    )
    embedding = np.asarray(embedding)

    name, _ = os.path.splitext(f)
    es.append((name, embedding))

if not es:
    print("There are no photos")
    sys.exit(1)

with open(os.path.join(config.ASSETS_DIR, "embeddings.pickle"), "wb") as f:
    pickle.dump(es, f)
