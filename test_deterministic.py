#-*- coding:utf-8 -*-

# MIT License
# 
# Copyright (c) 2020 Youngsam Kim

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time
import sys
import numpy as np
import importlib
import argparse
import pickle
import random
import imageio

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers

AUTOTUNE = tf.data.experimental.AUTOTUNE

def main(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #filter INFO
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no        
    
    ## tf-deterministic 
    if args.deterministic:                
        os.environ['TF_DETERMINISTIC_OPS'] = '1'    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    ## setting gpu memory
    gpus = tf.config.experimental.list_physical_devices('GPU') #return 1 GPU because of 'CUDA_VISIBLE_DEVICES'
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True) # dynamic memory only growing
        except RuntimeError as e:
            print(e)
    
    nrof_classes = 10
    weight_decay = 1e-4
    batch_size = 256

    ## building a model0
    img_size = 112
    img_inputs = K.Input(shape=(img_size, img_size, 3), name="img_inputs")
    x = K.layers.Conv2D(filters=64, kernel_size=[3,3], strides=1)(img_inputs)
    # x = K.layers.DepthwiseConv2D(kernel_size=[3,3], strides=1, depth_multiplier=1,
    #             padding='same', activation='relu', use_bias=False,
    #             kernel_initializer=K.initializers.HeNormal(seed=2020),
    #             kernel_regularizer=K.regularizers.L2(weight_decay))(x)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dropout(0.5, seed=2020)(x)
    embeddings = K.layers.Dense(64, activation=None)(x)
    base_model = K.Model(inputs=img_inputs, outputs=embeddings) # feature extration model
    
    #classfication head
    logit_layer = Logits(nrof_classes, weight_decay=weight_decay)
    logits = logit_layer(base_model.output)

    train_model = K.Model(inputs=[base_model.input], outputs=[embeddings, logits])
    # train_model.summary()

    # Instantiate an optimizer.
    # optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    optimizer = K.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.1)
    train_model.compile(optimizer=optimizer)

    # Instantiate a loss function.
    loss_fn = K.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Prepare the training dataset.    
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
    x_train = np.array((60000, img_size, img_size, 3)) #temporal
    y_train = y_train[:60000]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    epochs = 10
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                embs, logits = train_model((x_batch_train, y_batch_train), training=True)  # Logits for this minibatch
                logits = tf.nn.softmax(logits)

                # Compute the loss value for this minibatch.
                ce_loss = loss_fn(y_batch_train, logits)

                total_loss = tf.add_n([ce_loss] + train_model.losses)

            grads = tape.gradient(total_loss, train_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, train_model.trainable_variables))

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(total_loss))
                )
                # print("Seen so far: %s samples" % ((step + 1) * 64))
                
            ## debug code
            # if step == 200:
            #     with open('debug_train{}.pkl'.format(args.gpu_no), 'wb') as f:
            #         pickle.dump((x_batch_train, y_batch_train, embs, embs, grads, train_model.trainable_variables), f)
            #     exit()

    with open('debug_train{}.pkl'.format(args.gpu_no), 'wb') as f:
        pickle.dump((x_batch_train, y_batch_train, 0, embs, grads, train_model.trainable_variables), f)
        

class Logits(K.layers.Layer):
    def __init__(self, nrof_classes, weight_decay=0.0):
        super(Logits, self).__init__()
        self.nrof_classes = nrof_classes
        self.weight_decay = weight_decay

    def build(self, input_shape):        
        """
        Args:
            input_shape = emb_shape
        """

        self.W = tf.Variable(name='W', dtype=tf.float32,
                            initial_value=K.initializers.HeNormal(seed=2020)(shape=(input_shape[-1], self.nrof_classes)))
        self.b = tf.Variable(name='b', dtype=tf.float32,
                            initial_value=tf.zeros_initializer()(shape=[self.nrof_classes]))        
        #weight regularization
        self.add_loss(K.regularizers.L2(self.weight_decay)(self.W))

    def call(self, inputs):
        return tf.matmul(inputs, self.W) + self.b

    def get_config(self):
        config = super(Logits, self).get_config()
        config.update({"nrof_classes": self.nrof_classes,
                       "weight_decay": self.weight_decay,
                      })
        return config

    def compute_output_shape(self, input_shape):        
        return (None, self.nrof_classes)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu_no', type=str, help='Set visible GPU.', default='0')
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=333)
    parser.add_argument('--deterministic',
        help='Enable deterministic training', action='store_true')
    
    return parser.parse_args(argv)  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
