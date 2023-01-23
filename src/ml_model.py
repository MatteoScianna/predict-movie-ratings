#%%
#If you do not have a module, pip nstall
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import utils
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import sigmoid
from tensorflow import keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import csv
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
import argparse
import os
import json

#%% Load useful dictionaries

#DICTIONARY WITH MOVIE FEATURES
DIR = "/Users/Matteo/Desktop/Magistrale/Computational_Linguistics/imsdb_download_all_scripts/"
with open(DIR+"dict_features.json") as json_file: 
    dict_features = json.load(json_file)

#%%

#DICTIONARY WITH MOVIES IMDB IDs

with open(DIR+"dict_movies_id.json") as json_file:
    dict_movies_id = json.load(json_file)

#%%

#DICT WITH ALL WORDS AND CORRESPONDENT IDs

with open(DIR+"dict_words_id.json") as json_file:
    dict_words_id = json.load(json_file)

# %%

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def get_data(using_revenues, dict_features,word_index): #dict_features = dict_movie_features_final, #word_index = dict_words_id
    
    # Create dictionaries with imdbIDs and correspondent frequence vectors
    freq_vecs = {}
    l=[]
    for key, value in dict_features.items():
        try:
          freq_vecs[key] = [word[0] for word in value["top250words"]]
        except:
           l.append(key)

    #create arrays (here still lists) with other movies features
    data = [] # Variables we want to predict (Gross and Rating)
    freq_data = [] #What obtained from freq_vecs
    dir_data = [] #Complexity indices (feature "sintactic")
    metadata = [] #Numeric and categorical features (Language*, Genre**, Budget, Year of Release, Runtime)
    #*Language needs after to be numeralized, with genres we have problems up to now
    i = 0
    j = 0
    for key, value in dict_features.items():
        try:
            metadatacat = []
            #metadatacat.append(value["Languages"][0])
            metadatacat.append(value["Genres"])
            metadatacat.append(int(value["Budget"]))
            metadatacat.append(int(value["Year"]))
            metadatacat.append(int(value["Runtimes"][0]))
            metadatacat = flatten(metadatacat)
            metadata.append(metadatacat)
            i+=1
            print("Movie "+str(i)+": "+ key + "appended to metadata")
            rating = float(value["Rating"])
            gross = np.log(float(value["Gross"])+0.1)
            freq_data.append(freq_vecs[key])
            data.append([gross,rating])
            j+=1
            print("Movie "+str(j)+": "+ key + "appended to data")
            dir_data.append(value["sintactic"])
        except:
            pass


    #shuffle freq_data, dir_data, data and metadata arrays in the same way
    np.random.seed(1)
    index_map = np.arange(len(data))
    np.random.shuffle(index_map)

    rand_metadata = np.array([metadata[i] for i in index_map])
    rand_data = np.array([data[i] for i in index_map])
    rand_freqs = np.array([freq_data[i] for i in index_map]).astype(float).T
    rand_dir_data = np.array([dir_data[i] for i in index_map]).astype(float).T

    # Select the input array
    X = np.array(rand_data).T.astype(float)

    # Select optimization target based on command line args
    Y = np.array(rand_data)[np.newaxis,:,-1].astype(float)  
    if using_revenues:
        Y = np.array(rand_data)[np.newaxis,:,-2].astype(np.float)

    # Seperation of dataset into train, dev, and test sets
    #X_train = X[:,:(math.floor(.8 * X.shape[1]))]
    metadata_train = rand_metadata[:(math.floor(.8 * X.shape[1]))]
    freq_train = rand_freqs[:,:(math.floor(.8 * X.shape[1]))]
    dir_train = rand_dir_data[:,:(math.floor(.8 * X.shape[1]))]
    y_train = Y[0,:(math.floor(.8 * X.shape[1]))]

    #X_dev = X[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    metadata_dev = rand_metadata[(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    freq_dev = rand_freqs[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    dir_dev = rand_dir_data[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    y_dev = Y[0,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]

    #X_test = X[:,(math.floor(.9 * X.shape[1])):]
    metadata_test = rand_metadata[(math.floor(.9 * X.shape[1])):]
    freq_test = rand_freqs[:,(math.floor(.9 * X.shape[1])):]
    dir_test = rand_dir_data[:,(math.floor(.9 * X.shape[1])):]
    y_test = Y[0,(math.floor(.9 * X.shape[1])):]

    return word_index, y_train, freq_train, dir_train, metadata_train, freq_dev, dir_dev, y_dev, metadata_dev, freq_test, dir_test, y_test, metadata_test

def load_data(metadata, freq, direction): #Metadata = [Lang, Genre, Budget, Year, Runtime]
    return {"numeric": metadata[:,32:], "categorical":  metadata[:,0:32], "most_common_words": freq.T, "direction_subset": direction.T}

def setup_numeric(inputs, features):
    numeric_input = keras.Input(shape=(3,), name="numeric") #3 depending on the number of numeric features
    # Add inputs and features 
    inputs.append(numeric_input)
    features.append(numeric_input)

def create_word_embedding(word_index):
    
    # Load word embeddings 
    path_to_glove_file = "/Users/Matteo/Downloads/glove.6B.100d.txt"
    embeddings_index = {}
    with open(path_to_glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    # Prepare embedding matrix for our corpus
    num_tokens = len(word_index) + 2
    print(len(word_index))
    embedding_dim = 100
    hits = 0
    misses = 0
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for i, word in word_index.items():
        embedding_vector = embeddings_index.get(i)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[word] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix, num_tokens, embedding_dim

def setup_categorical(inputs, features):
    categorical_input = keras.Input(shape=(32,), name="categorical") #1 depends on number of categorical features
    #create categorical features
    categorical_features = Dense(128, activation='linear', use_bias=False)(categorical_input)
    # Add inputs and features 
    inputs.append(categorical_input)
    features.append(categorical_features)

def setup_frequency(inputs, features, embedding_layer):
    frequency_input = keras.Input(shape=(250,), name="most_common_words")
    
    freq_features = embedding_layer(frequency_input)
    freq_features = Flatten()(freq_features)

     #preprocess word embeddings
    freq_features = BatchNormalization()(freq_features)
    freq_features = Dense(1024, kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(freq_features)
    freq_features = Dense(256, kernel_regularizer=regularizers.l2(l2_param),  activation='tanh')(freq_features)

    inputs.append(frequency_input)
    features.append(freq_features)

def setup_direction(inputs, features, embedding_layer):
    direction_input = keras.Input(shape=(19,), name="direction_subset")

    dir_features = embedding_layer(direction_input)
    
    print(dir_features)

    #preprocess word embeddings
    dir_features = layers.LSTM(1, return_sequences=False)(dir_features)
    print(dir_features)

    inputs.append(direction_input)
    features.append(dir_features)

def score_out(X):
       return 10 * sigmoid(X)

def revenue_out(X):
   return 23 * sigmoid(X)
#%% Here we finally create the model 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--rev' """THIS IS FOR THE PREDICTED VARIABLE""", action='store_true') #false if you want it true (conterintuitive lol)
    parser.add_argument('--num' """THIS IS FOR THE NUMERICAL PREDICTORS""", action='store_true')
    parser.add_argument('--cia' """THIS IS FOR THE TOP WORDS PREDICTORS""", action = "store_false")
    parser.add_argument('--cat """THIS IS FOR THE CATEGORICAL PREDICTORS"""', action="store_true")
    parser.add_argument('--dir' """THIS IS FOR THE COMPLEXITY INDICES PREDICTORS""", action='store_false')
    args, unknown = parser.parse_known_args()

    word_index, y_train, freq_train, dir_train, metadata_train, freq_dev, dir_dev, y_dev, metadata_dev, freq_test, dir_test, y_test, metadata_test = get_data(args.rev, dict_features,dict_words_id) #CHANGE

    l2_param = 0.005
    inputs = []
    features = []

    if args.rev:
        utils.get_custom_objects().update({'custom_activation': Activation(revenue_out)})
    else: 
        utils.get_custom_objects().update({'custom_activation': Activation(score_out)})

    if args.num:
        setup_numeric(inputs, features)
    if args.cat:
        setup_categorical(inputs, features)

    if args.cia or args.dir:
        embedding_matrix, num_tokens, embedding_dim = create_word_embedding(dict_words_id)
        embedding_layer = Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )
    if args.cia:
        setup_frequency(inputs, features, embedding_layer)
    if args.dir:
        setup_direction(inputs, features, embedding_layer)

    #concatenate all features
    if len(features) > 1:
        x = layers.concatenate(features)
    else: 
        x = features[0]

    #fully connected network 
    x = BatchNormalization()(x)
    x = Dense(1024,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = Dense(512,  kernel_regularizer=regularizers.l2(l2_param), activation='relu')(x)
    x = Dense(256,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = Dense(128,  kernel_regularizer=regularizers.l2(l2_param), activation='relu')(x)
    x = Dense(64,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = Dense(32,  kernel_regularizer=regularizers.l2(l2_param), activation='relu')(x)
    outputs = Dense(1,  kernel_regularizer=regularizers.l2(l2_param), activation='custom_activation')(x)

    # Create model
    optimizer = keras.optimizers.Adam(learning_rate=0.005)
    model = keras.Model(inputs, outputs, name="Script2Score")

    # We use early stopping to help prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_squared_error",
        min_delta=0,
        patience=3000,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    # Compile model
    model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=["mean_squared_error"])
    # Train model
    history = model.fit(
                load_data(metadata_train, freq_train, dir_train),
                y_train.T,
                batch_size=50,
                epochs=50,
                callbacks=[early_stopping],
                verbose=1,
                validation_data=(load_data(metadata_dev,freq_dev, dir_dev), y_dev.T)
            )

    print(model.predict(load_data(metadata_dev, freq_dev, dir_dev)))

    # Summary of neural network, saved in 'model' folder
    model.summary()

    score = model.evaluate(load_data(metadata_dev, freq_dev, dir_dev), y_dev.T)
    print("Dev set results:")
    print(score)

    score = model.evaluate(load_data(metadata_test, freq_test, dir_test), y_test.T)
    print("Test set results:")
    print(score)

    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    model.save('model')

    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model MSE')
    plt.ylabel('mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
