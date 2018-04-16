import pandas as pd
import numpy as np

import re

import tensorflow

from tensorflow.contrib import keras
from keras.layers import Input, Dense, Embedding, merge,Convolution2D, MaxPooling2D, Dropout
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential,Input
from keras.layers import LSTM, SimpleRNN, Activation, Flatten, BatchNormalization

from writtenToSpoken import *

def splitText(text):
	splits = text.split()
	for i in range(len(splits)):
		if (splits[i][-1] == '.'
			or splits[i][-1] == ','
			or splits[i][-1] == '?'
			or splits[i][-1] == '!'
			or splits[i][-1] == ':'
			or splits[i][-1] == ';'):
			splits[i] = splits[i][:-1]
	return splits
	
def text_to_df(text):
    data = splitText(text)
    return pd.DataFrame(data, columns=["before"])

def get_encodings(data_set,length_of_encodings):
    average_token_length = length_of_encodings
    x_data = []
    spacing = 0
    for x in data_set['before'].values:
        x_row = np.ones(average_token_length, dtype=int) * spacing
        for xi, i in zip(list(str(x)), np.arange(average_token_length)):
            x_row[i] = ord(xi)
        x_data.append(x_row)
    x_data=np.asarray(x_data)
    return x_data

def create_model():
    sequence_length = 10 #= x_data.shape[1]
    vocabulary_size = 240
    embedding_dim = 10
    filter_sizes = [3,4,5]
    num_filters = 512
    drop = 0.5
    nb_epoch = 2
    batch_size = 30
	
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)
    conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', 
                           init='normal', activation='relu', dim_ordering='tf')(reshape)
    conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid',
                           init='normal', activation='relu', dim_ordering='tf')(reshape)
    conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid',
                           init='normal', activation='relu', dim_ordering='tf')(reshape)
    maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1),
                             border_mode='valid', dim_ordering='tf')(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1),
                             border_mode='valid', dim_ordering='tf')(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1),
                             border_mode='valid', dim_ordering='tf')(conv_2)
    merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
    flatten = Flatten()(merged_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(output_dim=16, activation='softmax')(dropout)
    model = Model(input=inputs, output=output)
    return model

# NORMALIZATION BEGINS
def plain(x):
    try:
        out=plain_words[x]
        return out
    except:
        return x
		
def normalize_text(text):
	df = text_to_df(text)

	encodings = get_encodings(df, 10)

	model = create_model()
	model.load_weights("weights.001-0.9943.hdf5")

	predicted_df = model.predict(encodings)

	indices = np.argmax(predicted_df, axis=1)

	label_names = ['PLAIN', 'PUNCT', 'DATE', 'LETTERS', 'CARDINAL', 'VERBATIM', 'DECIMAL', 'MEASURE', 'MONEY', 'ORDINAL', 'TIME', 'ELECTRONIC', 'DIGIT', 'FRACTION', 'TELEPHONE', 'ADDRESS']

	df2 = [label_names[idx] for idx in indices]
	df["class"] = df2

	df_plain = df[df['class']=="PLAIN"]
	df_plain['after']=df_plain['before'].apply(lambda x: plain(x))

	df_punct = df[df['class']=="PUNCT"]
	df_punct['after']=df_punct['before']

	df_cardinal = df[df['class']=="CARDINAL"]
	output=[]
	for i in range(df_cardinal.shape[0]):
		output.append(cardinal(df_cardinal.iloc[i,0]))
	df_cardinal['after']=output

	df_date = df[df['class']=="DATE"]
	output=[]
	for i in range(df_date.shape[0]):
		output.append(date(df_date.iloc[i,0]))
	df_date['after']=output

	df_letters = df[df['class']=="LETTERS"]
	output=[]
	for i in range(df_letters.shape[0]):
		output.append(letters(df_letters.iloc[i,0]))
	df_letters['after']=output

	df_verbatim = df[df['class']=="VERBATIM"]
	output=[]
	for i in range(df_verbatim.shape[0]):
		output.append(verbatim(df_verbatim.iloc[i,0]))
	df_verbatim['after']=output

	df_money = df[df['class']=="MONEY"]
	output=[]
	for i in range(df_money.shape[0]):
		output.append(money(df_money.iloc[i,0]))
	df_money['after']=output

	df_measure = df[df['class']=="MEASURE"]
	output=[]
	for i in range(df_measure.shape[0]):
		output.append(measure(df_measure.iloc[i,0]))
	df_measure['after']=output

	df_electronic = df[df['class']=="ELECTRONIC"]
	output=[]
	for i in range(df_electronic.shape[0]):
		output.append(electronic(df_electronic.iloc[i,0]))
	df_electronic['after']=output

	df_telephone = df[df['class']=="TELEPHONE"]
	output=[]
	for i in range(df_telephone.shape[0]):
		output.append(telephone(df_telephone.iloc[i,0]))
	df_telephone['after']=output

	df_decimal=df[df['class']=="DECIMAL"]
	output=[]
	for i in range(df_decimal.shape[0]):
		output.append(decimal(df_decimal.iloc[i,0]))
	df_decimal['after']=output

	df_digit=df[df['class']=="DIGIT"]
	output=[]
	for i in range(df_digit.shape[0]):
		output.append(digit(df_digit.iloc[i,0]))
	df_digit['after']=output

	output=[]
	df_ordinal=df[df['class']=="ORDINAL"]
	for i in range(df_ordinal.shape[0]):
		output.append(ordinal(df_ordinal.iloc[i,0]))
	df_ordinal['after']=output

	output=[]
	df_fraction=df[df['class']=="FRACTION"]
	for i in range(df_fraction.shape[0]):
		output.append(fraction(df_fraction.iloc[i,0]))
	df_fraction['after']=output

	output=[]
	df_time=df[df['class']=="TIME"]
	for i in range(df_time.shape[0]):
		output.append(time(df_time.iloc[i,0]))
	df_time['after']=output

	output=[]
	df_address=df[df['class']=="ADDRESS"]
	for i in range(df_address.shape[0]):
		output.append(address(df_address.iloc[i,0]))
	df_address['after']=output

	results = df_plain.append(df_address).append(df_cardinal).append(df_date).append(df_digit).append(df_electronic).append(df_letters).append(df_decimal).append(df_measure).append(df_money).append(df_punct)
	results = results.append(df_ordinal).append(df_fraction).append(df_telephone).append(df_time).append(df_verbatim)
	results = results.sort_index()

	arr_temp = np.array(results["after"])
	spoken_text = " ".join(map(str, arr_temp))
	
	return spoken_text
