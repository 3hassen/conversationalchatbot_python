import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
from tensorflow.python.framework import ops
import random
import json
import pickle

words = []
labels = []
docs_x = []
docs_y = []

#nltk.download('punkt')

with open("C:/Users/hasse/Desktop/projects/conversational chatbot/intents.json") as f:
    data = json.load(f)

#print (data["intents"])
try:
    with open ("data.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)
except:   
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            docs_x.append(word)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            
    words = [stemmer.stem(w.lower()) for w in words]
    words = sorted(list(set(words)))
    #print(words)
    
    labels = sorted(labels)
    #print(labels)
    
    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag = []
        
        word = [stemmer.stem(w) for w in doc]
        
        for w in words:
            if w in word:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    training = np.array(training)
    output = np.array(output)
    with open ("data.pickle", "wb") as file:
        pickle.dump((words, labels, training, output), file) 

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try: 
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    
def bag_of_words (s, words):
    bag= [0 for _ in range (len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for h in s_words:
        for i, w in enumerate(words):
            if w == h:
                bag[i] = 1
                
    return np.array(bag)

def chat():
    print("start talking with the bot !")
    while True:
        a = input("You: ")
        if a.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(a, words)])
        #return(results)
        results_index = np.argmax(results)
        tag = labels[results_index]
        #print(tag)
        
        #if results[results_index] > 0.7:     
        for r in data["intents"]:
            if r['tag'] == tag:
                response = r['responses']
                        
        print(random.choice(response))
        #else:
            #print("I didn't get that, try again bro")
        
chat()

