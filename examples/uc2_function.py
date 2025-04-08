def fire_presence_detection(im: list[list[int]]) -> str | bool:
    import sys
    import traceback

    try:
        
        import os
        f = open('/dev/null','w')
        sys.stdout = f
        
        import cv2
        import tflearn #type: ignore
        import numpy as np
        
        from tflearn.layers.core import input_data, dropout, fully_connected #type: ignore
        from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d, global_avg_pool #type: ignore
        from tflearn.layers.normalization import local_response_normalization, batch_normalization #type: ignore
        from tflearn.layers.merge_ops import merge #type: ignore
        from tflearn.layers.estimator import regression #type: ignore

        import tensorflow as tf #type: ignore
        
        tf.compat.v1.reset_default_graph()
        
        # Numpy Array cannot be passed to function due to serialization issues in Cognit
        im = np.asarray(im, dtype=np.uint8)
        # cv2.imwrite('.image.png', im)
        # image = cv2.VideoCapture('.image.png')

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
        if(training):
            pool5_7_7 = dropout(pool5_7_7, 0.4)
        loss = fully_connected(pool5_7_7, 2,activation='softmax')

        # if training then add training hyperparameters

        if(training):
            network = regression(loss, optimizer='momentum',
                                loss='categorical_crossentropy',
                                learning_rate=0.001)
        else:
            network = loss;

        model = tflearn.DNN(network, checkpoint_path='inceptiononv1onfire',
                            max_checkpoints=1, tensorboard_verbose=2)
        
        model.load(os.path.join("/root/userapp/InceptionV1-OnFire", "inceptiononv1onfire"),weights_only=True)
    #-----------------------------------------------MODEL LOAD---------------------------------------
        
        # load video file from first command line argument

        # ret, frame = image.read()
        
        # re-size image to network input size and perform prediction

        small_frame = cv2.resize(im, (rows, cols), cv2.INTER_AREA)

        # perform prediction on the image frame which is:
        # - an image (tensor) of dimension 224 x 224 x 3
        # - a 3 channel colour image with channel ordering BGR (not RGB)
        # - un-normalised (i.e. pixel range going into network is 0->255)

        output = model.predict([small_frame])

        # label image based on prediction

        return round(output[0][0]) == 1 # equiv. to 0.5 threshold in [Dunnings / Breckon, 2018],  [Samarth/Bhowmik/Breckon, 2019] test code
    except Exception as error:
        exc_info = sys.exc_info()
        sExceptionInfo = ''.join(traceback.format_exception(*exc_info))
        exc_type, exc_value, exc_context = sys.exc_info()
        bug = str(sExceptionInfo)
        return bug