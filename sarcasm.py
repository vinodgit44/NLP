#import libraries
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow import keras


with open("../input/sarcasm-data/sarcasm.json") as f:
    file1=json.load(f)
print(type(file1))
headline=[]
sarcas=[]
#split json file data into two lists
for items in file1:
    headline.append(items["headline"])
    sarcas.append(items["is_sarcastic"])

train_data=headline[0:20000]
test_data=headline[20000:]
train_label=sarcas[0:20000]
test_label=sarcas[20000:]

tokenizer=Tokenizer(num_words=10000,oov_token="<oov>")#oov token is provided to new
tokenizer.fit_on_texts(train_data) #make tokens
wordindex=tokenizer.word_index    #see tokens
print(wordindex)


train_seq=tokenizer.texts_to_sequences(train_data)  #apply tokens to sequence of data
#padding train data

train_pad=pad_sequences(train_seq,maxlen=100,padding='post',truncating='post')

#fitting token to test data
test_seq=tokenizer.texts_to_sequences(test_data)
test_pad=pad_sequences(test_seq,maxlen=100,padding="post",truncating="post")

#convert data into numpy arrays
train_pad=np.array(train_pad)
test_pad=np.array(test_pad)
train_label=np.array(train_label)
test_label=np.array(test_label)

#creating model
model=tf.keras.Sequential([
    tf.keras.layers.Embedding(10000,16,input_length=100),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])


#compiling model
model.compile(optimizer="adamax",loss="binary_crossentropy",metrics=['accuracy'])
model.summary()

#fit test and validation data into models
model.fit(train_pad,train_label,epochs=30,validation_data=(test_pad,test_label),verbose=2)
# model.save('asd.h5')#save model

# model = keras.models.load_model('asd.h5') #loads model
#test new data
sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
print(model.predict(padded))