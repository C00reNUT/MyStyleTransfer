#this code is my own copy
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import time
import functools
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models

content_path = "./content/test.jpeg"
style_path = "./style/test.jpg"

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def load_and_process_img(path_to_img): #loads image
    img = Image.open(path_to_img)

    img = kp_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img): #not my code
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content(generated_activation, content_activation):
    return tf.reduce_mean(tf.square(content_activation - generated_activation))

def gram_matrix(layer):
    channels = int(layer.shape[-1])
    a = tf.reshape(layer, [-1, channels]) #flatten
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32) #normalizes for how many layers you have

def style_loss(generated_activation, style_activation):
    generated = gram_matrix(generated_activation)
    style = gram_matrix(style_activation)
    return tf.reduce_mean(tf.square(generated - style)) #this will take the difference of each

def load_model(): #this part is not my code
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False  # we're simply using the CNN as a "loss function"

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]  # basically outputs for comparison
    model_outputs = style_outputs + content_outputs

    return models.Model(vgg.input, model_outputs)  # creates the model using keras structure

def big_loss_function(content, style, output):
    pass

def get_content_and_style_features(model, image, style):
    image_features = model(image)
    style_outputs = model(style)

    style_features = style_outputs[:num_style_layers]
    image_features = image_features[num_style_layers:] #allright, we have the stuff!

    return style_features, image_features
