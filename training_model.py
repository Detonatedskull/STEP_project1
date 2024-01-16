# We are going to design a simple sequential training model using the following modules 
# Module 1: json-> this is to import and read json files especially the intents.json 
# Module 2: pickle-> A very basic library in pythhon in order manipulate binary files and stor data 
# module 3: random-> to give different responses based on the intents file 
# module 4: numpy-> to manipulate arrays and numerical data to be concise
# module 5: tensorflow and keras-> To train mainframe neural networks and other machine learning based operations
# module 6: nltk -> One of the most important libraries which will help us to actuate natural language processing

# This .py file will only train the ML model 
# importing the required modules
import random
import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Activation, Dropout
import seaborn as sns 
import matplotlib.pyplot as plt 
  
#The WordNetLemmatizer from NLTK is used to convert words to their base or dictionary forms (lemmas).
lemmatizer = WordNetLemmatizer()
  
# data loading, reading the intents file for the chatbots initial training dataset
intents = json.loads(open("intents.json").read())

# processing intent data
# creating empty lists to store data
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # separating words from patterns
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # and adding them to words list
          
        # associating patterns with respective tags
        documents.append(((word_list), intent['tag']))
  
        # appending the tags to the class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
  
# storing the root words or lemma
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
words = sorted(set(words))
  
# saving the words and classes list to binary files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

output_empty = [0] * len(classes)

training = []
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
          
    # making a copy of the output_empty
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)

# Convert the training data to a suitable format
train_x = np.array([bag for bag, _ in training])  # Extracting bag of words
train_y = np.array([output_row for _, output_row in training])  # Extracting output rows


# creating a Sequential machine learning model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ),
                activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), 
                activation='softmax'))
  
# compiling the model
sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True) #we are not using the traditional SGD due to recent deprication, we have used legacy optimisers for using the same
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=120, batch_size=5, verbose=1)

# model training loss visualisation
plt.figure(figsize=(10, 6))
plt.plot(hist.history['loss'], label='loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# same for training accuracy
plt.figure(figsize=(10, 6))
plt.plot(hist.history['accuracy'], label='accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# saving the model
model.save("chatitout.h5", hist)

print("Chatbot has been successfully trained!!")
