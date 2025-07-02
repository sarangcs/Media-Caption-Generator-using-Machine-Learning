from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#import cv2
import os
import warnings
from nltk.translate.bleu_score import sentence_bleu
warnings.filterwarnings("ignore")
from rouge import Rouge 

import nltk.translate.meteor_score
#nltk.download('wordnet')

img_path = "FLICKR/Flickr8k_Dataset/23445819_3a458716c1.jpg"
def extract_features(filename, model):
        try:
            image = Image.open(filename)

        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
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
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text



max_length = 32
tokenizer = load(open("C:/Users/saran/generator/tokenizer.p","rb"))
model = load_model("C:/Users/saran/models/model_56.h5")
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)

print("\n\n")
(description)
plt.imshow(img)

res=description.split(' ', 1)[1] 


 
#split string
spl_string = res.split()
 
#remove the last item in list
rm = spl_string[:-1]
 
#convert list to string
Caption = ' '.join([str(elem) for elem in rm])
 
#print string
print(Caption)

#BLEU METRIC
spl_string = img_path.split('/')
cap=spl_string[-1]
    #print(str(cap))

f = open("C:/Users/saran/generator/descriptions.txt", "r")

l=[]
with open("C:/Users/saran/generator/descriptions.txt") as openfile:
        for line in openfile:
            for part in line.split('#'):
                if cap in part:
                    #print (part)
                    l.append(part.split())
                
#print(l)
    
l[0].pop(0)
l[1].pop(0)
l[2].pop(0)
l[3].pop(0)
l[4].pop(0)
#print(l)

    

reference = l

candidate = Caption.split()
#print(candidate)
print("\n")
print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))
 
print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))

#ROUGE METRIC
#ROUGE METRIC
listToStr1 = ' '.join(map(str, l[0])) 
listToStr2 = ' '.join(map(str, l[1])) 
listToStr3 = ' '.join(map(str, l[2])) 
listToStr4 = ' '.join(map(str, l[3])) 
listToStr5 = ' '.join(map(str, l[4])) 

  
#print(listToStr)  

reference =listToStr1






#rouge = Rouge()
#scores = rouge.get_scores(hypothesis, reference)
print("\n\n")
#print(scores)

#print("Meteor:")
#print(nltk.translate.meteor_score.meteor_score([listToStr1,listToStr2,listToStr3,listToStr4,listToStr5],Caption))