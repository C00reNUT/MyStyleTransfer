#this code is my own (mostly)
#the only non-original things are processing functions

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
import pickle
import os

content_path = "./content/test.JPG"
style_path = "./style/test.jpg"

content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

STYLE_CONTRIB = 0.01/num_style_layers #feel free to change these values
CONTENT_CONTRIB = 1000/num_content_layers

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

def content_loss(generated_activation, content_activation):
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

def big_loss_function(model, style_features, image_features, output): #we don't make style/image features in house
    #because they are never going to change
    output_features_style, output_features_image = get_output_features(model, output)
    content_loss_value = 0
    style_loss_value = 0

    assert len(output_features_style) == len(style_features)
    assert len(output_features_image) == len(image_features)

    for i in range(len(output_features_image)):
        content_loss_value += content_loss(output_features_image[i], image_features[i])

    for i in range(len(output_features_style)):
        style_loss_value += style_loss(output_features_style[i], style_features[i])

    big_loss = CONTENT_CONTRIB * content_loss_value + STYLE_CONTRIB * style_loss_value

    return big_loss, content_loss_value, style_loss_value

def get_output_features(model, output):
    model_outputs = model(output)

    output_features_style = model_outputs[:num_style_layers]
    output_features_image = model_outputs[num_style_layers:]

    return output_features_style, output_features_image

def get_style_and_content_features(model, image, style): #returns all feature extraction from the models
    image_features = model(image)
    style_outputs = model(style)

    style_features = style_outputs[:num_style_layers]
    image_features = image_features[num_style_layers:] #allright, we have the stuff!

    return style_features, image_features


def run_style_transfer(content_path, style_path, num_iterations, load):
    model = load_model()
    for layer in model.layers:
        layer.trainable = False

    if load:
        try:
            with open("Best_Image.pkl", 'rb') as fo:
                object = pickle.load(fo, encoding='bytes')
        except:
            print("You are trying to restore an image that doesn't exist!")
            quit()

        assert len(object) == 1, "there should only be one loaded image"
        print(np.shape(object))
        output_image = tf.Variable(initial_value =object, dtype = tf.float32)
    else:
        output_image = tf.Variable(initial_value=load_and_process_img(content_path), dtype=tf.float32)
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)
    style_features, content_features = get_style_and_content_features(model, content_image, style_image)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1) #hyperparmeters are not mine

    best_loss, best_img = float('inf'), None #this are initial values


    norm_means = np.array([103.939, 116.779, 123.68]) #this section is not mine
    min_vals = -norm_means
    max_vals = 255 - norm_means
    print("I'm starting!")
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            all_loss, content_loss_value, style_loss_value = big_loss_function(model, style_features, content_features, output_image)

        gradients = tape.gradient(all_loss, output_image)
        optimizer.apply_gradients([(gradients, output_image)])

        clipped = tf.clip_by_value(output_image, min_vals, max_vals) #some black magic with bounding

        output_image.assign(clipped)

        if all_loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = all_loss
            best_img = deprocess_img(output_image.numpy())

        if i % int(num_iterations/20) == 0:
            img = Image.fromarray(best_img.astype(np.uint8), "RGB")
            img.save("evolution/" + str(i) + ".jpg") #saves images
            try:
                os.remove("Best_Image.pkl")
            except:
                pass
            dbfile = open("Best_Image.pkl", "ab")
            pickle.dump(best_img, dbfile)
            dbfile.close()

        print('Iteration: {}'.format(i))
        print('Total loss: {:.4e}, '
              'style loss: {:.4e}, '
              'content loss: {:.4e}, '
              .format(all_loss, style_loss_value, content_loss_value))

    return best_img, best_loss

def main():
    num_eval = int(input("How many iterations?"))
    status = input("restore? (t/f)")
    if status == 't':
        bool_status = True
    else:
        bool_status = False
    best_img, best_loss = run_style_transfer(content_path, style_path, num_eval, bool_status)
    img = Image.fromarray(best_img.astype(np.uint8), "RGB")
    img.save("combined.jpg")

if __name__ == "__main__":
    main()