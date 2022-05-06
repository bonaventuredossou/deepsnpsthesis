# -*- coding: utf-8 -*-
"""Integrated Gradients

From the paper [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) and its [GitHub](https://github.com/CVxTz/IntegratedGradientsPytorch)

The goal is to compute the feature importances of our model on test samples

Interesting tutorial: https://medium.com/@madhubabu.adiki/integrated-gradients-for-natural-language-processing-from-scratch-c81c50c5bc4d

Load our testing dataset and select one sample per category `cancer` and `non-cancer`
"""

import tensorflow as tf
import json
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np

datapath = '/CTCF Binding Sites/'
import os
import pandas as pd

training = pd.read_csv(datapath + 'training_data_filtered.csv')
validation = pd.read_csv(datapath + 'validation_data_filtered.csv')
testing = pd.read_csv(datapath + 'test_data_filtered.csv')

init_training_len = len(training)
init_validation_len = len(validation)
init_testing_len = len(testing)

dataset = pd.concat([training, validation, testing])
SEED = 1234
dataset.head()

def mapping_to_lower_interval(a, b, c, d, t):
  return c + ((d-c)/(b-a))*(t-a)

def retrieve_initial_value(a, b, c, d, t):
  # t == f(x)
  return (((t-c)*(b-a))/(d-c)) + a

min_dist = min(dataset.min_distance.tolist())
max_dist = max(dataset.min_distance.tolist())

min_chr_pos = min(dataset.CHR_POS.tolist())
max_chr_pos = max(dataset.CHR_POS.tolist())

# here we will also transform the SNPS, to the interval [0, embedding_size)

min_snps_pos = min(dataset.SNPS.tolist())
max_snps_pos = max(dataset.SNPS.tolist())

new_min_dist = 0
new_max_dist = 2000

new_min_pos = 0
new_max_pos = 5000
new_max_pos_1 = 2048

new_min_snps = 0
new_max_snps = 2048

def transform_min_distance(value):
  return mapping_to_lower_interval(min_dist, max_dist, new_min_dist, new_max_dist, value)

def transform_chr_position(value):
  return mapping_to_lower_interval(min_chr_pos, max_chr_pos, new_min_pos, new_max_pos, value)

def transform_chr_position_2(value):
  return mapping_to_lower_interval(new_min_pos, new_max_pos, new_min_pos, new_max_pos_1, value)

def transform_snps(value):
  return mapping_to_lower_interval(min_snps_pos, max_snps_pos, new_min_snps, new_max_snps, value)

dataset['CHR_POS'] = dataset['CHR_POS'].apply(transform_chr_position) # original -> [0, 5000]
dataset['CHR_POS'] = dataset['CHR_POS'].apply(transform_chr_position_2) # [0, 5000] -> [0, 2048]

dataset['min_distance'] = dataset['min_distance'].apply(transform_min_distance)
dataset['SNPS'] = dataset['SNPS'].apply(transform_snps)

testing = dataset[init_validation_len:init_validation_len+init_testing_len]

"""Load the model"""

modelpath = '/path/model/deep_snps_model.h5'
from tensorflow import keras
model = keras.models.load_model(modelpath)

embedding_layer = model.get_layer('embedding')

# build new model with all layers after embedding layer
new_model = keras.Sequential()
for layer in model.layers[2:]:
    new_model.add(layer)

"""Linearly interpolate from baseline vector to the sample vector"""

def interpolate(baseline, vector, m_steps):
    """ Linearly interpolate the sample vector
    (embedding layer output)"""

    # Generate m_steps intervals for integral_approximation() below.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    delta = vector - baseline
    texts = baseline + alphas_x * delta
    return texts

"""Compute gradients of the model output for the interpolated vectors with respect to the vectors, and particularly the prediction index of the true class."""

def compute_gradients(t, target_class_idx):
    """ compute the gradient with respect to embedding layer output """
    with tf.GradientTape() as tape:
      tape.watch(t)
      probs = new_model(t)[:, target_class_idx]
    grads = tape.gradient(probs, t)
    return grads

feature_importances = {"correct_predictions":[], "wrong_predictions":[]}

def get_feature_importance(dataframe):
  print("Computing features importances on the test dataset")
  labels = dataframe["is_cancer"].tolist()
  snps = dataframe["SNPS"].tolist()
  contexts = dataframe["CONTEXT"].tolist()
  chr_ids = dataframe["CHR_ID"].tolist()
  chr_poss = dataframe["CHR_POS"].tolist()
  min_dists = dataframe["min_distance"].tolist()

  for snp, context, chr_id, chr_pos, min_dist, label in zip(snps, contexts, chr_ids, chr_poss, min_dists, labels):
    input_vector = [snp, context, chr_id, chr_pos, min_dist]
    label_input = label
    print(input_vector, label_input)
    """Select a baseline: the paper suggests using a zero embedding vector"""
    example = tf.constant(input_vector)
    example_embed = embedding_layer(example)
    baseline_embed = tf.zeros(shape=tf.shape(example_embed))

    n_steps = 50
    interpolated = interpolate(baseline_embed,
                                           example_embed,
                                           n_steps)

    # sample label is the true class of the sample
    path_gradients = compute_gradients(interpolated, label_input)

    # sum the grads of the interpolated vectors
    all_grads = tf.reduce_sum(path_gradients, axis=0) / n_steps

    # mulitply grads by (input - baseline); baseline is zero vectors
    x_grads = tf.math.multiply(all_grads, example_embed)

    # sum all gradients across the embedding dimension
    igs = tf.reduce_sum(x_grads, axis=-1).numpy()

    features = ["SNPS", "CONTEXT", "CHR_ID", "CHR_POS", "min_distance"] 
    
    importances = [features[i] for i in tf.argsort(igs, -1, 'DESCENDING')]

    input_ = tf.constant([input_vector])
    predicted_class = np.argmax(model.predict(input_), axis=1)[0]

    if predicted_class == label_input:
      feature_importances["correct_predictions"].append(importances)
    else:
      feature_importances["wrong_predictions"].append(importances)


get_feature_importance(testing)

with open("feature_importances.json", 'w') as outfile:
    json.dump(feature_importances, outfile)
