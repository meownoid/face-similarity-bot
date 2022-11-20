import io
import logging
import multiprocessing
import os
import pickle
import sys
from operator import itemgetter

import dlib
import numpy as np
from PIL import Image
from telegram import Bot
from telegram.ext import Filters, MessageHandler, Updater

import config

logging.basicConfig(level=config.LOG_LEVEL)

if not os.path.exists(config.LANDMARKS_PATH):
    print(f"{config.LANDMARKS_PATH} does not exist")
    sys.exit(1)

if not os.path.exists(config.MODEL_PATH):
    print(f"{config.MODEL_PATH}  does not exist")
    sys.exit(1)

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(config.LANDMARKS_PATH)
face_recognition_model = dlib.face_recognition_model_v1(config.MODEL_PATH)

with open(os.path.join(config.ASSETS_DIR, "embeddings.pickle"), "rb") as f:
    star_embeddings = pickle.load(f)


def handle_photo(update, _):
    global bot

    message = update.message
    photo = message.photo[~0]

    with io.BytesIO() as fd:
        file_id = bot.get_file(photo.file_id)
        file_id.download(out=fd)
        fd.seek(0)

        image = Image.open(fd)
        image.load()
        image = np.asarray(image)

    face_detects = face_detector(image, 1)

    if not face_detects:
        bot.send_message(chat_id=update.message.chat_id, text="no faces")

    face = face_detects[0]
    landmarks = shape_predictor(image, face)
    embedding = face_recognition_model.compute_face_descriptor(image, landmarks)
    embedding = np.asarray(embedding)

    ds = []

    for name, emb in star_embeddings:
        distance = np.linalg.norm(embedding - emb)
        ds.append((name, distance))

    best_match, best_distance = min(ds, key=itemgetter(1))

    bot.send_message(
        chat_id=update.message.chat_id,
        text=f"you look exactly like *{best_match}*",
        parse_mode="Markdown",
    )


bot = Bot(token=config.TOKEN)
updater = Updater(bot=bot, workers=multiprocessing.cpu_count())
dispatcher = updater.dispatcher
photo_handler = MessageHandler(Filters.photo, handle_photo)
dispatcher.add_handler(photo_handler)

updater.start_polling()
updater.idle()
