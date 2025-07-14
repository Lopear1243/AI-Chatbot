# chatbot.py
import random
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pickle

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load data
with open("intents.json", encoding="utf-8") as f:
    intents = json.load(f)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list):
    if intents_list:
        tag = intents_list[0]["intent"]
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "Sorry, I didn't understand that. Can you try again?"
