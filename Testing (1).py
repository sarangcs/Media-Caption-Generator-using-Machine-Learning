
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import cv2
import os
import warnings
from gtts import gTTS
import pygame
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

warnings.filterwarnings("ignore")
pygame.mixer.init()  # Initialize pygame mixer for audio playback

VORI = input("Enter Input v for video captioning / i for image captioning:")


def extract_features(filename, model):
    try:
        image = Image.open(filename)

    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension are correct")
        return None
    
    image = image.resize((299, 299))
    image = np.array(image)

    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


if VORI == "i":
    image_name = input("Enter image name:")
    img_path = "C:/Users/saran/generator/" + image_name

    if not (img_path.endswith('.jpg') or img_path.endswith('.jpeg')):
        print("ERROR: Invalid image format. Only .jpg and .jpeg formats are supported.")
    else:
        max_length = 32
        tokenizer = load(open("C:/Users/saran/generator/tokenizer.p", "rb"))
        model = load_model("C:/Users/Saran/generator/models/model_29.h5")
        xception_model = Xception(include_top=False, pooling="avg")

        photo = extract_features(img_path, xception_model)
        if photo is not None:
            img = Image.open(img_path)

            description = generate_desc(model, tokenizer, photo, max_length)
            print("\n\n")

            res = description.split(' ', 1)[1]

            spl_string = res.split()

            rm = spl_string[:-1]

            Caption = ' '.join([str(elem) for elem in rm])

            print(Caption)

            language = 'en'

            myobj = gTTS(text=Caption, lang=language, slow=False)

            myobj.save("Caption.mp3")

            pygame.mixer.music.load("Caption.mp3")
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)


if VORI == "v":
    video_name = input("Enter video name:")
    vidcap = cv2.VideoCapture(video_name + '.mp4')

    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite(os.path.join("C:/Users/Saran/generator/FRAMES", "image" + str(count) + ".jpg"), image)
        return hasFrames

    sec = 0
    frameRate = 30.0
    count = 1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)

    max_length = 32
    tokenizer = load(open("C:/Users/saran/generator/tokenizer.p", "rb"))
    model = load_model("C:/Users/Saran/generator/models/model_29.h5")
    xception_model = Xception(include_top=False, pooling="avg")
    i = 1
    with os.scandir("C:/Users/Saran/generator/FRAMES") as entries:
        for entry in entries:
            path = "C:/Users/Saran/generator/FRAMES" + "/" + entry.name
            photo = extract_features(path, xception_model)
            if photo is not None:
                img = Image.open(path)
                description = generate_desc(model, tokenizer, photo, max_length)
                print("\n\n")

                res = description.split(' ', 1)[1]

                spl_string = res.split()

                rm = spl_string[:-1]

                Caption = ' '.join([str(elem) for elem in rm])

                print(Caption)

                language = 'en'

                myobj = gTTS(text=Caption, lang=language, slow=False)

                myobj.save("Caption" + str(i) + ".mp3")

                pygame.mixer.music.load("Caption" + str(i) + ".mp3")
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

                i = i + 1

    images = []
    for img_path in glob.glob("C:/Users/Saran/generator/FRAMES/*.jpg"):
        images.append(mpimg.imread(img_path))