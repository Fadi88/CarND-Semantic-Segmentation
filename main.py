#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
from moviepy.editor import VideoFileClip
import numpy as np
import scipy

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return w1, keep, layer_3_out, layer_4_out, layer_7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function


    #https://blog.playment.io/wp-content/uploads/2018/02/fcn_arch_vgg16-700x327.png
    l2_reg = 1e-3
    conv_1_1 = tf.layers.conv2d(vgg_layer7_out , num_classes , 1 , padding="same" , 
                                kernel_regularizer= tf.contrib.layers.l2_regularizer(l2_reg))

    #deconv 1 layer + skip connection
    deconv_1 = tf.layers.conv2d_transpose(conv_1_1 , num_classes , 4 , (2,2) , padding='same' ,
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(l2_reg))

    deconv_1_skip = tf.layers.conv2d(vgg_layer4_out , num_classes , 1 , padding='same',
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(l2_reg))

    deconv_1_skipped = tf.add(deconv_1 , deconv_1_skip)

    #deconv 2 layer + skip conncetion
    deconv_2 = tf.layers.conv2d_transpose(deconv_1_skipped , num_classes , 4 , (2,2) , padding='same' ,
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(l2_reg))

    deconv_2_skip = tf.layers.conv2d(vgg_layer3_out , num_classes , 1 , padding='same',
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(l2_reg))

    deconv_2_skipped = tf.add(deconv_2 , deconv_2_skip)

    #deconv 3 layer no skip as per the paper
    deconv_3 = tf.layers.conv2d_transpose(deconv_2_skipped , num_classes , 16 , (8,8) , padding='same' ,
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(l2_reg))

    return deconv_3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # labels = tf.reshape(correct_label, (-1,num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_last_layer, labels=correct_label))

    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    for epoch in range(epochs):
        t0 = time.time()
        print('Epoch ' , str(epoch) , ' :')
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009})
        print("epochs was : " , time.time() -t0 , ' and final loss was : ' , loss)
    pass
tests.test_train_nn(train_nn)

def video_cb(img):
    image_shape = (160, 576)
    im_softmax = sess.run([tf.nn.softmax(logits)],{keep_prob: 1.0, image_pl: [img]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
    
        # TODO: Train NN using the train_nn function
        epochs = 50
        batch_size = 5
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video
        '''
        in_clip = VideoFileClip("")
        out_clip = in_clip.fl_image(video_cb)
        out_clip.write_videofile("", audio=False)
        '''


if __name__ == '__main__':
    run()
