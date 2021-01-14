# Importing Libraries
import ktrain
import os
import numpy as np
import tensorflow as tf
from ktrain import text

# Data Preprocessing
dataset = tf.keras.utils.get_file(fname='aclImdb_v1.tar.gz', 
                                  origin='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                                  extract=True)
IMDB_DATADIIR = os.path.join(os.path.dirname(dataset), 'aclImdb')

print(os.path.dirname(dataset))
print(IMDB_DATADIIR)

# Create training and Test sets
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(datadir=IMDB_DATADIIR,
                                                                       classes=['pos', 'neg'],
                                                                       maxlen=500,
                                                                       train_test_names=['train', 'test'],
                                                                       preprocess_mode='bert')

# BERT Model
Bert = text.text_classifier(name='bert', train_data=(x_train, y_train),
                            preproc=preproc)

# Training BERT model
learner = ktrain.get_learner(model=Bert,
                             train_data=(x_train, y_train),
                             val_data=(x_test, y_test),
                             batch_size=6)

learner.fit_onecycle(lr=2e-5,
                     epochs= 1)
