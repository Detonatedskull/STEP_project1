# STEP_project1
NLP Project with very basic communication skills
This is chatbot version 1.0 

-------------running the chatbot---------------------------
Follow the following steps to run the chatbot correctly 
1. Open training_model.py and run the program make sure you have all the libraries mentioned in the program installed on your computer 
2. You can optionally view the accuracy and loss graphs by seaborn
3. Make sure of the presence of these files after training inside your directory:
    i. chatitout.h5
    ii. intents.json
    iii. classes.pkl
    iv. main.py
    v. words.pkl
    if not present rerun the training_model.py module
4. Open main.py and run the program, make sure you have all the libraries mentioned in the program installed on your device

------------questions you can ask for better communication (as the same is limited)-------------------------
Q1. Hello
Q2. How are you?
Q3. Tell me a joke 
Q4. Who inspires you?
Q5. Whats your favorite sport
etc. you can find more questions in the intents.json file

-------------------MORE INFORMATION ABOUT THE CHATBOT-------------------------
Read OUTPUTS.docx for output screenshots

The chatbot uses the following modules:

random
json
pickle
numpy
nltk
tensorflow 
keras.models : Sequential
nltk.stem : WordNetLemmatizer
keras.layers : Dense, Activation, Dropout
keras.optimizers : SGD, legacy
seaborn
matplotlib.pyplot
google-search-python
time
datetime

json-> this is to import and read json files especially the intents.json 
pickle-> A very basic library in pythhon in order manipulate binary files and stor data 
random-> to give different responses based on the intents file 
numpy-> to manipulate arrays and numerical data to be concise
tensorflow and keras-> To train mainframe neural networks and other machine learning based operations
nltk -> One of the most important libraries which will help us to actuate natural language processing

The chatbot model is saved as HDF5 model rather than the keras model as there were version fluctuations in the program that i had to counter
We have used keras optimizers and legacy operators to connect keras and tensorflow together to disevaluate these fluctuations
This has not reduced the efficiency of the chatbot. 
The training accuracy and loss data have been visualised during the training, you can refer to the same in this zip file

This chatbot has limited functionality due to reduced complexity for a intermediate level AI programmer. 
Please feel free to contact pillaikiran88@gmail.com if encountered with any form of errors
