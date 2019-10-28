#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 13:56:00 2019

@author: jeffreyscruggs
"""

import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.keras.callbacks


mnist = tf.keras.datasets.fashion_mnist

(training_images,training_labels), (test_images,test_labels) = mnist.load_data()

plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')<0.2):
      print("\nReached 80% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks=myCallback()
training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,activation='relu'),
                                    tf.keras.layers.Dense(256,activation='relu'),
                                    tf.keras.layers.Dense(10,activation = 'softmax' )])

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images,training_labels,epochs=20,callbacks=[callbacks])

print(model.evaluate(test_images, test_labels))

