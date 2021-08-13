import keras
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, Cropping2D
import tensorflow as tf

def build_unet_2015(input_layer):   

    conv1 = Conv2D(64, (3, 3), activation="relu", padding="valid")(input_layer)    
    conv1 = Conv2D(64, (3, 3), activation="relu", padding="valid")(conv1)

    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="valid")(pool1)
    conv2 = Conv2D(128, (3, 3), activation="relu", padding="valid")(conv2)
    
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="valid")(pool2)
    conv3 = Conv2D(256, (3, 3), activation="relu", padding="valid")(conv3)
    
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    # 16 -> 8
    conv4 = Conv2D(512, (3, 3), activation="relu", padding="valid")(pool3)
    conv4 = Conv2D(512, (3, 3), activation="relu", padding="valid")(conv4)
    
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    convm = Conv2D(1024, (3, 3), activation="relu", padding="valid")(pool4)
    convm = Conv2D(1024, (3, 3), activation="relu", padding="valid")(convm)

    deconv4 = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(convm)
    
    cropsize4 = int((conv4.shape[1]-deconv4.shape[1])/2)
    conv4crop = Cropping2D(cropping=((cropsize4, cropsize4), (cropsize4, cropsize4)))(conv4)
    uconv4 = concatenate([deconv4, conv4crop])
    uconv4 = Dropout(0.5)(uconv4)
    
    uconv4 = Conv2D(512, (3, 3), activation="relu", padding="valid")(uconv4)
    uconv4 = Conv2D(512, (3, 3), activation="relu", padding="valid")(uconv4)

    deconv3 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(uconv4)
    
    cropsize3 = int((conv3.shape[1]-deconv3.shape[1])/2)
    conv3crop = Cropping2D(cropping=((cropsize3, cropsize3), (cropsize3, cropsize3)))(conv3)
    uconv3 = concatenate([deconv3, conv3crop])
    uconv3 = Dropout(0.5)(uconv3)
    
    uconv3 = Conv2D(256, (3, 3), activation="relu", padding="valid")(uconv3)
    uconv3 = Conv2D(256, (3, 3), activation="relu", padding="valid")(uconv3)

    deconv2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(uconv3)
    
    cropsize2 = int((conv2.shape[1]-deconv2.shape[1])/2)
    conv2crop = Cropping2D(cropping=((cropsize2, cropsize2), (cropsize2, cropsize2)))(conv2)
    uconv2 = concatenate([deconv2, conv2crop])
    uconv2 = Dropout(0.5)(uconv2)
    
    uconv2 = Conv2D(128, (3, 3), activation="relu", padding="valid")(uconv2)
    uconv2 = Conv2D(128, (3, 3), activation="relu", padding="valid")(uconv2)

    deconv1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(uconv2)
    
    cropsize1 = int((conv1.shape[1]-deconv1.shape[1])/2)
    conv1crop = Cropping2D(cropping=((cropsize1, cropsize1), (cropsize1, cropsize1)))(conv1)
    uconv1 = concatenate([deconv1, conv1crop])    
    uconv1 = Dropout(0.5)(uconv1)
    
    uconv1 = Conv2D(64, (3, 3), activation="relu", padding="valid")(uconv1)
    uconv1 = Conv2D(64, (3, 3), activation="relu", padding="valid")(uconv1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer

@tf.function
def val_step(input_layer, label, model, val_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(input_layer, training=False)
        loss = keras.losses.BinaryCrossentropy(from_logits = False)
        loss_value = loss(label, logits)
        val_acc_metric.update_state(label, logits)
        return loss_value
    
@tf.function
def train_step(input_layer, label, model, optimizer, train_acc_metric):     
    
    with tf.GradientTape() as tape:
        logits = model(input_layer, training=True)
        loss = keras.losses.BinaryCrossentropy(from_logits = False)
        loss_value = loss(label, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))   
    
    train_acc_metric.update_state(label, logits)
    
    return loss_value 
    

