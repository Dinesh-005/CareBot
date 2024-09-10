import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents2.json').read())
words = pickle.load(open('words.pkl','rb'))
labels = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    print(return_list)
    return return_list



#getting chatbot response
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


#Creating GUI with tkinter

import tkinter
from tkinter import *
from PIL import Image, ImageTk
import tkinter.messagebox as tkMessageBox

def Exit():
    result=tkMessageBox.askquestion('System', 'Are you sure you want to exit?', icon="warning")
    if result == 'yes':
        base.destroy()
        exit()

def show_about():
    tkMessageBox.showinfo("About", "The AI Medical Chatbot project, developed by Dinesh, is an innovative application designed to assist users with healthcare-related queries through an intuitive chat interface. The project leverages artificial intelligence, specifically natural language processing (NLP) and machine learning algorithms, to provide real-time medical advice and support.")
        
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("AI health consultant")
base.geometry("1200x655")
image = Image.open("main.jpg")

tk_image = ImageTk.PhotoImage(image)

label = Label(base, image=tk_image)
label.pack()

h=Label(base, text="AI MEDICAL CHATBOT", fg="#952186", bg="#9DC1E0",font=("arial 30 bold")).place(x=420,y=25)
b=Label(base, text="AI Medical Symptom Checker Chatbot", fg="black",bg="#729BC7",font=("arial 12 bold")).place(x=900,y=600)

ChatLog = Text(base, bd=5, bg="#8BD0EC", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=3,
                    bd=5, bg="#DF8843", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

EntryBox = Text(base, bd=5, bg="white",width="29", height="3", font="Arial")

scrollbar.place(x=825,y=100, height=386)
ChatLog.place(x=450,y=100, height=386, width=370)
EntryBox.place(x=575, y=490, height=90, width=265)
SendButton.place(x=450, y=490, height=90)
menubar = Menu(base)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Exit", command=Exit)
menubar.add_cascade(label="File", menu=filemenu)
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=show_about)
menubar.add_cascade(label="Help", menu=helpmenu)
base.config(menu=menubar)

base.mainloop()
