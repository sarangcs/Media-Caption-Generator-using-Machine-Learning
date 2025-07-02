import tkinter as tk
from tkinter import Label, Button, filedialog, messagebox
from PIL import ImageTk, Image
import cv2
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import warnings
from gtts import gTTS
import pygame
import glob

warnings.filterwarnings("ignore")
pygame.mixer.init()

# Load tokenizer and model
tokenizer = load(open("C:/Users/saran/generator/tokenizer.p", "rb"))
model = load_model("C:/Users/saran/models/model_56.h5")
xception_model = Xception(include_top=False, pooling="avg")
max_length = 32

# Initialize Tkinter window
top = tk.Tk()
top.geometry('800x600')
top.title('Caption Generator')
top.configure(background='#CDCDCD')

# Add a main title
title_label = tk.Label(top, text="Media Caption Generator", font=('Arial', 20, 'bold'), background='#CDCDCD')
title_label.pack(pady=20)

# Label for displaying image or video preview
image_label = tk.Label(top, background='#CDCDCD')
image_label.pack(expand=True)

# Global variables for storing image preview
img_preview = None
video_preview = None

# Function to extract features from an image
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        messagebox.showerror("Error", "Couldn't open image! Make sure the image path and extension are correct.")
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

# Function to generate a description for an image
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
    return in_text[5:-3]

# Function to get the word for a given ID from the tokenizer
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to preview an image
def preview_image(filename):
    global img_preview
    img = Image.open(filename)
    img = img.resize((400, 300), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    img_preview = img_tk

# Function to preview a video
def preview_video(filename):
    global video_preview
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (400, 300))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    video_preview = img_tk
    cap.release()

# Function to open and caption an image or video based on the file type
def open_media():
    filename = filedialog.askopenfilename(title="Select Media File", filetypes=[("Image Files", "*.jpg *.jpeg"), ("Video Files", "*.mp4")])
    if filename:
        if filename.lower().endswith(('.jpg', '.jpeg','.webp')):
            preview_image(filename)
            caption_image(filename)
        elif filename.lower().endswith('.mp4'):
            preview_video(filename)
            caption_video(filename)
        else:
            messagebox.showerror("Error", "Unsupported file format!")

# Function to open and caption an image
def caption_image(filename):
    photo = extract_features(filename, xception_model)
    if photo is not None:
        description = generate_desc(model, tokenizer, photo, max_length)
        messagebox.showinfo("Image Caption", description)
        generate_audio(description)

# Function to open and caption a video
def caption_video(filename):
    temp_folder = "temp_frames"
    os.makedirs(temp_folder, exist_ok=True)

    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    count = 1
    first_caption_generated = False  # Flag to track the first caption generated
    while success and not first_caption_generated:
        cv2.imwrite(os.path.join(temp_folder, "frame" + str(count) + ".jpg"), image)
        success, image = vidcap.read()
        count += 1

        # Process only the first frame for caption generation
        if count == 2:  # Assuming you want to process the second frame (change if needed)
            img_path = os.path.join(temp_folder, "frame1.jpg")
            photo = extract_features(img_path, xception_model)
            if photo is not None:
                description = generate_desc(model, tokenizer, photo, max_length)
                print(description)
                messagebox.showinfo("First Video Caption", description)
                generate_audio(description)
                first_caption_generated = True  # Set the flag to True

    # Clean up temporary frames if the directory exists
    if os.path.exists(temp_folder):
        for file in glob.glob(f"{temp_folder}/*.jpg"):
            os.remove(file)
        os.rmdir(temp_folder)

# Function to generate audio from a caption
def generate_audio(description):
    language = 'en'
    audio_path = 'captionspeech.mp3'
    myobj = gTTS(description, lang=language, slow=False)
    myobj.save(audio_path)
    
    # Attempt to play the audio
    try:
        os.system(f"start {audio_path}")
        print("Audio played successfully.")
    except Exception as e:
        print("Error playing audio:", e)

# Create button for media input
btn_open_media = tk.Button(top, text="INPUT MEDIA", command=open_media)
btn_open_media.pack(padx=10, pady=5)

# Run the Tkinter event loop
top.mainloop()
