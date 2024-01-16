#importing important modules
#Almost all modules have similar functionalities as explained in training_model.py 
#A few ne modules have been included such as time, datetime and googlesearch 
#These modules have been included in the main.py file as they aid in some of the functionalities of the chatbot model 
import random
import json
import pickle
import numpy as np
import nltk
import time 
import datetime
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from googlesearch import search

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatitout.h5')

#chatbot important functions  

#cleaning up, tokenisation and othre crucial functions
def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)
                      for word in sentence_words]
    return sentence_words

#bag of words
def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#response prediction 
def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res)
               if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
        return return_list

#repsonse capture and comparison with the intents.json file
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

#google_search functionality
def search_system(user_input):
    if user_input[:6].lower() == 'search':
        query = message[6:]
        print("Fetching results from the web: ")
        search_results = search(query)
        
        response = "Here are some search results:\n"
        for idx, result in enumerate(search_results, start=1):
            response += f"{idx}. {result}\n"
        
        print(response)
        pass
    
#weather system 
def weather_system(user_input):
    if  any(check in user_input for check in ["weather","huidity","temprature", "what's the weather in"]):
        query = user_input
        city = input("The city you want the weather for: ")
        search_results = search(f"{query} {city}")
        
        response = "Weather results from google..."
        for idx, result in enumerate(search_results, start=1):
            response += f"{idx}. {result}\n"
        
        print(response)
        pass

#timer system
def timer_system(user_input):
    if any(check in user_input.lower() for check in ["set a timer","timer","start a countdown","countdown"]):
        seconds=int(input("Time: "))
        while seconds > 0:
            minutes, secs = divmod(seconds, 60)
            timeformat = '{:02d}:{:02d}'.format(minutes, secs)
            print("\rTime remaining: {}".format(timeformat), end="\r")
            time.sleep(1)
            seconds -= 1

        print("\rTime's up!\n")
    
    elif any(check in user_input.lower() for check in ["what's the time?","time","date","what's the date"]):
        print("current date and time are: ",datetime.datetime.now())



print("Chatbot is up!")
print("\n\n\n\n****************WELCOME TO CHAT-IT-OUT V1.0*****************\n\n\n")
print("****** FUNCTIONALITIES ***********")
print("I am a general purpose chatbot. My capabilities are : \n 1. I can chat with you. Try asking me for jokes or riddles! \n 2. Ask me the date and time \n 3. I can google search for you. Use format search: <query> \n 4. I can get the present weather for any city. Use format search: weather <city name> \n 5. I can set a timer for you. Enter 'set a timer: minutes to timer' \n 6.use command \'exit\' or key combination ctrl+C to exit \n\n For suggestions to help me improve, send an email to pillaikiran88@gmail.com .Thank you!!\n")
print("\n\n\n*****START CHATTING*******\n\n\n")

message=""
while message.lower()!="exit":
    message = input("user: ")
    search_system(message)
    weather_system(message)
    timer_system(message)
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res,"\n\n")
