import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}



DEBUG = False


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
    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    vgg_input_tensor_name = 'image_input:0'    
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    if DEBUG:
    	print("***********************************************************")
    	print(input_tensor)
    	print(keep_prob)
    	print(layer3)
    	print(layer4)
    	print(layer7)
    	print("***********************************************************")


    return input_tensor, keep_prob, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def conv_1x1(layer, num_outputs, layer_name = "default_conv"):
    kernel_size = 1

    return tf.layers.conv2d(layer, num_outputs, kernel_size, strides = (1,1), name = layer_name)


def upsample(layer, num_classes, layer_name = "default_upsample", kernel = 4, stride = (2,2)):
    """
    """
       
    return tf.layers.conv2d_transpose(layer, num_classes, kernel, strides = stride, padding = 'SAME', name = layer_name, kernel_initializer=tf.truncated_normal_initializer(stddev = 0.01) )


def skip(layer_back, layer_front, layer_name = "skip"):
	return tf.add(layer_back, layer_front, name = layer_name)



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    #ENCODER
    layer7_1by1 =  conv_1x1(vgg_layer7_out, num_classes,"layer7_1by1")


    #DECODER
    layer_7_trans = upsample(layer7_1by1, num_classes, layer_name = "layer7_trans")

    # 1x1 convolution of the 4th layer 
    layer4_1by1 = conv_1x1(vgg_layer4_out, num_classes, layer_name = "layer_4_1by1")

    #Add element wise the 4 layer with the output of the 7th layer
    skip_4th = skip(layer4_1by1, layer_7_trans, layer_name = "skip_4th")

    #upsample the addition of the 4th and 7th layer
    layer_47_trans = upsample(skip_4th, num_classes, layer_name = "add47_trans")

				# 1x1 convolution of the 3th layer 
    layer3_1by1= conv_1x1(vgg_layer3_out, num_classes, "layer_3_1by1")

    #Add element wise the 3 layer with the output of the last layer
    skip_3th = skip(layer3_1by1, layer_47_trans, layer_name = "skip_3th" )

				#upsample the addition of the 3th and last layer
    output = upsample(skip_3th, num_classes, layer_name = "output", kernel = 16, stride = (8,8))




    return output
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

    logits = tf.reshape(nn_last_layer,(-1,num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

 #    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(correct_label,1))
 
	# # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# # print("Accuracy: ", accuracy)





    return (logits, optimizer, cross_entropy_loss)
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

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for epoch in range(epochs):
    	print("Epoch {}...".format(epoch))
    	for images, labels in get_batches_fn(batch_size):
    		_ , loss= sess.run([train_op, cross_entropy_loss], feed_dict = {input_image: images, correct_label:labels, keep_prob:0.5, learning_rate:0.00005})


    print("Loss: {}".format(loss))
    

tests.test_train_nn(train_nn)


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

    #HYPERPARAMETERS
    epochs = 10
    batch_size = 64
    

    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        #get_batches_test = helper.gen_test_output(os.path.join(data_dir, 'data_road/testing'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        learning_rate = tf.placeholder(tf.float32, name="alpha")

        correct_label = tf.placeholder(tf.float32,(None, None, None, num_classes), name="y")
        
        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3, layer4, layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)


        
		#print("Test accuracy %g"%accuracy.eval(feed_dict={x: X_test, y_: y_test, keep_prob: 1.0}))

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        helper.save_inference_samples(runs_dir, data_dir,sess, image_shape, logits, keep_prob, input_image)
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    print("Running...")
    run()
