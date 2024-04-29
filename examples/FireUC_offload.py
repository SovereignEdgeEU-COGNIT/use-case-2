# This is needed to run the example from the cognit source code
# If you installed cognit with pip, you can remove this

image = "Test_images/fire3.png"

def fire_presence_detection(im):
    
    try:
        import cv2
        import tflearn

        from tflearn.layers.core import input_data, dropout, fully_connected
        from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool
        from tflearn.layers.normalization import local_response_normalization, batch_normalization
        from tflearn.layers.merge_ops import merge
        from tflearn.layers.estimator import regression

        import os
        import tensorflow as tf
        
        tf.compat.v1.reset_default_graph()
        
        cv2.imwrite('.image.png', im)
        image = cv2.VideoCapture('.image.png')

        rows = 224
        cols = 224

        x = rows
        y = cols
        training=False
        enable_batch_norm=True

        # Build network as per architecture in [Dunnings/Breckon, 2018]

        network = input_data(shape=[None, y, x, 3])

        conv1_7_7 = conv_2d(network, 64, 5, strides=2, activation='relu', name = 'conv1_7_7_s2')

        pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
        pool1_3_3 = local_response_normalization(pool1_3_3)

        conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
        conv2_3_3 = conv_2d(conv2_3_3_reduce, 128,3, activation='relu', name='conv2_3_3')

        conv2_3_3 = local_response_normalization(conv2_3_3)
        pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')

        inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')

        inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
        inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
        inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
        inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
        inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
        inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

        # merge the inception_3a__
        inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

        inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
        inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
        inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
        inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
        inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
        inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
        inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

        #merge the inception_3b_*
        inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

        pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
        inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
        inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
        inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
        inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
        inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
        inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

        inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')

        pool5_7_7 = avg_pool_2d(inception_4a_output, kernel_size=5, strides=1)

        network = loss;

        model = tflearn.DNN(network, checkpoint_path='inceptiononv1onfire',
                            max_checkpoints=1, tensorboard_verbose=2)
        
        model.load(os.path.join("/root/FireUC/model", "inceptiononv1onfire"),weights_only=True)
    #-----------------------------------------------MODEL LOAD---------------------------------------
        
        # load video file from first command line argument

        ret, frame = image.read()
            

        # re-size image to network input size and perform prediction

        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

        # perform prediction on the image frame which is:
        # - an image (tensor) of dimension 224 x 224 x 3
        # - a 3 channel colour image with channel ordering BGR (not RGB)
        # - un-normalised (i.e. pixel range going into network is 0->255)

        output = model.predict([small_frame])

        # label image based on prediction

        if round(output[0][0]) == 1: # equiv. to 0.5 threshold in [Dunnings / Breckon, 2018],  [Samarth/Bhowmik/Breckon, 2019] test code
            return 1
        else:
            return 0
    except:
        return 2

import sys	
import time

sys.path.append(".")

from cognit import (
    EnergySchedulingPolicy,
    FaaSState,
    ServerlessRuntimeConfig,
    ServerlessRuntimeContext,
)

# Configure the Serverless Runtime requirements
sr_conf = ServerlessRuntimeConfig()
sr_conf.name = "Example Serverless Runtime"
sr_conf.scheduling_policies = [EnergySchedulingPolicy(50)]

# Configure "Nature" flavour to use FireUC image 
sr_conf.faas_flavour = "Nature"

# Request the creation of the Serverless Runtime to the COGNIT Provisioning Engine

try:
    # Set the COGNIT Serverless Runtime instance based on 'cognit.yml' config file
    my_cognit_runtime = ServerlessRuntimeContext(config_path="cognit.yml")

    # Perform the request of generating and assigning a Serverless Runtime to this Serverless Runtime context.
    ret = my_cognit_runtime.create(sr_conf)
except Exception as e:
    print("Error in config file content: {}".format(e))
    exit(1)

# Checks the status of the request of creating the Serverless Runtime, and sleeps 1 sec if still not available.
while my_cognit_runtime.status != FaaSState.RUNNING:
    time.sleep(1)

time.sleep(5)

print("COGNIT Serverless Runtime ready!")

import cv2

# select the image to analyze and import it
image = cv2.imread(image)

# perform the function offloading
result = my_cognit_runtime.call_sync(fire_presence_detection, image)

print("Offloaded function result", result)
print("Result", result.res)

# This sends a request to delete this COGNIT context.
my_cognit_runtime.delete()

print("COGNIT Serverless Runtime deleted!")
