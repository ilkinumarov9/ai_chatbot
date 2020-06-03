import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pickle

stemmer = LancasterStemmer()
nltk.download('punkt')


#loading data
with open("text.json") as txt:
    data = json.load(txt)

#adding try-catch block for not training model every time
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # defining lists which will use
    words = []
    labels = []
    documents = []
    documents_2 = []

    #looking each tag and pattern in dictionary, and appending
    #tokenized patterns to word list, then appending patterns them
    #selfees to one list, and tags to another list
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrd = nltk.word_tokenize(pattern)
            words.extend(wrd)
            documents.append(wrd)
            documents_2.append(intent["tag"])
    #we are appending tags to labels list if it is not already in it
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #using lowercase words for precision, and sorting unique words to list
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #we are appending 0 to output stream iinitially for length
    #of labels
    training = []
    output = []
    output_empty = [0 for l in range(len(labels))]

    #one hot encoding: adding words to bag of words every time,
    #creating bag of words
    for x,doc in enumerate(documents):
        bag = []
        wrd = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrd:
                bag.append(1)
            else:
                bag.append(0)

        output_row = output_empty[:]
        #in documents 2, we are storing tags(labels).
        #if we have that label on output, appending 1 to output
        output_row[labels.index(documents_2[x])] = 1
        #appending bag to training list, bag of words
        training.append(bag)
        output.append(output_row)

    #Converting to np arrays, because tflearn needs arrays
    training = np.array(training)
    output = np.array(output)

    #saving model
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)



tf.reset_default_graph() #get rid of previous settings
#we are getting same length inputs, so training is zero

#constructing architecture
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8) #fully connected layer is added
net = tflearn.fully_connected(net,8)
#output layer
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
#softmax means giving probabilities each neuron.
net = tflearn.regression(net) # applying regression


model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

#bag of words method
def bag_of_words(s,words):
    bag = [0 for l in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    #generating bag of words
    for sent in s_words:
        for i, w in enumerate(words):
            if w == sent:
                bag[i] = 1 # the word exists in sentence

    return np.array(bag)


def chat():
    print("Hi! Start talking with me. Tip: Type quit to quit :)")
    while True:
        i = input("Person: ")
        if i == "quit":
            print("Quitting from the chatbot....")

            break
        results = model.predict([bag_of_words(i,words)])
        res_index =np.argmax(results) # select higest probab node
        tag = labels[res_index]
        #open json, find the tag and respond with arbitrary resp

        for t in data["intents"]:
            if t['tag']==tag:
                response = t["responses"]
        print(random.choice(response))

chat()