
import os
import sys
import numpy as np

import scipy.misc
import tensorflow as tf
from PIL import Image
import build_vgg


OUTPUT_PATH = "D:\\Data Science Experiment\\CNN\\image_transformation\\"
STYLE_IMAGE_PATH = "D:\\Data Science Experiment\\CNN\\image_transformation\\style.jpg"
CONTENT_IMAGE_PATH = "D:\\Data Science Experiment\\CNN\\image_transformation\\content.jpg"

# SETTINGS
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
COLOR_CHANNELS = 3
NOISE_RATIO = 0.6
ITERATIONS = 1000

# parameters
alpha = 1
beta = 500



MEAN_VALUE = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
# 1x1x1x3 print(mean_value)


# the layers to use and the weight of evaluation
CONTENT_LAYERS = [('conv4_2', 1.)]
STYLE_LAYERS = [('conv1_1', 0.2), ('conv2_1', 0.2), ('conv3_1', 0.2), ('conv4_1', 0.2), ('conv5_1', 0.2)]



def add_noise_to_content(content_image, noise_ratio = NOISE_RATIO):
    noise = np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')
    # 1 x height x width x color_channels
    image = noise_ratio * noise + (1-noise_ratio) * content_image
    return image

# return an image of (1 x 1 x pixels x 3)
def load_image(path):
    image = Image.open(path)
    #image = image.convert("P")
    #image = image.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    #print(image.size)
    image_list = np.array(image.getdata())
    image = np.reshape(image_list, (1,) + (IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    #print(np.size(image))
    image = image - MEAN_VALUE
    return image

def save_image(path, image):
    image = image + MEAN_VALUE;
    """
    pixels = pixels[0][0]
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT))
    image.save(path + "output.jpg")
    """
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

############## algorithm functions

######## Compute the loss of the Content and input #########

def content_layer_loss(p, x):
    M = p.shape[1] * p.shape[2]
    N = p.shape[3]
    loss = (1. / (2 * N * M)) * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def content_loss_func(sess, net):

    layers = CONTENT_LAYERS
    total_content_loss = 0.0
    for layer_name, weight in layers:
        act = sess.run(net[layer_name])
        x = net[layer_name]
        total_content_loss += content_layer_loss(act, x) * weight

    total_content_loss /= float(len(layers))
    return total_content_loss

################# Compute the loss of Style and input #############
    
def gram_matrix(x, area, depth):

    x1 = tf.reshape(x, (area, depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g


def style_layer_loss(a, x):

    M = a.shape[1] * a.shape[2]
    N = a.shape[3]
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G - A), 2))

    return loss


def style_loss_func(sess, net):

    layers = STYLE_LAYERS
    total_style_loss = 0.0

    for layer_name, weight in layers:
      # When the input is the style image, this will 
      # compute the value of these activations.
        act = sess.run(net[layer_name])  
        
      # But x is just a net archetecture, not computed yet.
        x = net[layer_name]

        total_style_loss += style_layer_loss(act, x) * weight

    total_style_loss /= float(len(layers))

    return total_style_loss

#image = load_image("/home/fzy/CNNtest/style.jpg")
#save_image("/home/fzy/CNNtest/", image)


def main():
    net = build_vgg.build()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    content_image = load_image(CONTENT_IMAGE_PATH)
    style_image = load_image(STYLE_IMAGE_PATH)

    """
        Compute the cost of the content image
    """
    # Assign the content image to the input and 
    # RUN this assignment.
    sess.run([net['input'].assign(content_image)])
    
    # Compute relevant values and build the archetecture.
    cost_content = content_loss_func(sess, net)

    
    """
        compute the cost of the style image
    """
    sess.run([net['input'].assign(style_image)])
    
    cost_style = style_loss_func(sess, net)

    
    """
        Compute the total cost
    """

    total_loss = alpha * cost_content + beta * cost_style


    """
        initialize the input to optimize
    """
    init_img = add_noise_to_content(content_image)

    sess.run(tf.global_variables_initializer())
    sess.run(net['input'].assign(init_img))

    optimizer = tf.train.GradientDescentOptimizer(tf.constant(0.1))

    update_op = optimizer.minimize(total_loss, var_list=[net['input']])
    
    #update_op.run()
    sess.run(update_op)
    
    """
        get the result
    """
    mixed_img = sess.run(net['input'])

    save_image(OUTPUT_PATH + "output.jpg", mixed_img)


if __name__=='__main__':
    main()