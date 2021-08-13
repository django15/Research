import tensorflow as tf
import keras
import numpy as np
import os
import sys
import time
import PIL
from keras import optimizers
from keras import Model
from keras.layers import Input
from load_dataset import *
from model import *

'''Print versions of all the libraries used, to make sure user can replecate the code succesfully. 
print('Python', sys.version)
print('TensorFlow', tf.__version__) 
print('Keras', keras.__version__ ) 
print('numPy',np.__version__ )  
print('pandas', PIL.__version__)  
#print('pillow', pd.__version__)
print ('cpu compiler', tf.sysconfig.get_build_info()['cpu_compiler'])
print ('cuda compute capabilities', tf.sysconfig.get_build_info()['cuda_compute_capabilities'])
print ('cuda', tf.sysconfig.get_build_info()['cuda_version'])
print ('cudnn', tf.sysconfig.get_build_info()['cudnn_version'])
'''
train_data = 'Foot Ulcer Segmentation Challenge/train/images'
train_labels = 'Foot Ulcer Segmentation Challenge/train/labels'
val_data = 'Foot Ulcer Segmentation Challenge/validation/images'
val_labels = 'Foot Ulcer Segmentation Challenge/validation/labels'
test_data = 'Foot Ulcer Segmentation Challenge/test/images'
test_labels = 'Foot Ulcer Segmentation Challenge/test/labels'

print ('','#'*100, '\nNumber of Images in train, validation and test datasets are {}, {} and {} respectively.\n'.format(len(os.listdir(train_data)), len(os.listdir(val_data)), len(os.listdir(test_data)) ), '#'*100)

#uncomment to create tfrecord (for first time running the model.)
'''
write_tfrecord(train_data, train_labels, 'training')
write_tfrecord(val_data, val_labels, 'validation')
'''

#tfrecords location
training_data = 'tfrecord/training'
validation_data = 'tfrecord/validation'
#loading tfrecord in desired input format
training_data = load_tfrecord(training_data)
validation_data = load_tfrecord(validation_data)

#creating model
input_layer = Input(shape= (572, 572, 1))
output_layer = build_unet_2015(input_layer)
#print (output_layer)
model = keras.Model(inputs=input_layer, outputs=output_layer)
optimizer = keras.optimizers.SGD(learning_rate=0.005, momentum=0.95)

#creating accuracy metrices
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

batch_size = 64
n_epochs = 5


for epoch in range(n_epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()   
    
    training_data = training_data.shuffle(buffer_size = 1024)
    training_data = training_data.batch(1)
    validation_data = validation_data.shuffle(buffer_size = 1024)
    validation_data = validation_data.batch(batch_size)    
    itr_train = iter(training_data)
    itr_val = iter(validation_data)   
    
    try:
        step = 0
        while True:    
            t = time.time()
            train_image, _, train_label = itr_train.get_next()
            loss = train_step(train_image, train_label, model, optimizer, train_acc_metric)            
            step += 1
            t1 = time.time()
            if step % 1 == 0:
                print("Training loss at step %d in %.4f sec: %.4f" % (step, (t1-t),float(loss)))
    except tf.errors.OutOfRangeError:
        pass
    
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    train_acc_metric.reset_states()
    
    try:
        print ('validating training accuracy', '.'*100)
        while True:
            val_images, _, val_labels = itr_val.get_next()
            val_loss = val_step(val_images, val_labels, model, val_acc_metric)
            
    except tf.errors.OutOfRangeError:
        pass
    
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    
    print("Validation acc: %.4f" % (float(val_acc),))
    
    print("Time taken: %.2fs" % (time.time() - start_time))
    
    
    












