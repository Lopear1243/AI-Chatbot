import os
import json
import numpy as np
import random
import pickle
import nltk
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Download NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Load intents from intents.json
with open("intents.json") as file:
    data = json.load(file)

# Preprocessing
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    bag = [1 if w in word_patterns else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save("chatbot_model.h5")

# Utility functions
def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't understand that. Can you rephrase?"
    tag = intents_list[0]["intent"]
    for i in intents_json["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])

# Chat loop (for testing only, optional)
print("Chatbot is ready to talk! Type 'quit' to exit.")
while True:
    message = input("lopez: ")
    if message.lower() == "quit":
        break
    ints = predict_class(message, model)
    res = get_response(ints, data)
    print(f"Cafeteria Chatbot: {res}")


