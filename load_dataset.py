import tensorflow as tf
from PIL import Image
import os
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_image_binary(filename, label_flag):
    try:
        if label_flag:
            image = tf.io.read_file(filename)
            mask = tf.io.decode_png(image)
            mask = tf.image.resize(mask, (388,388), 
                                   method=tf.image.ResizeMethod.BILINEAR, 
                                   preserve_aspect_ratio=False,
                                   antialias=False)
            colors = [
                (0,0,0), # for background
                (255,255,255) #for ulcer
            ]
            one_hot_map = []
            for color in colors:
                class_map = tf.reduce_all(tf.equal(mask, color), axis=-1)
                one_hot_map.append(class_map)
            one_hot_map = tf.stack(one_hot_map, axis=-1)
            one_hot_map = tf.cast(one_hot_map, tf.float32)
            image = tf.argmax(one_hot_map, axis=-1)
        else:    
            image = Image.open(filename)
            image = image.convert('L')  
            image = image.resize((572,572), Image.ANTIALIAS)
        image = np.asarray(image, np.uint8)        
        image = image.tobytes()
        return image
    except Exception as e:
        pass

def write_tfrecord(data_loc, label_loc, tfrecord_name):
    tf_record_loc = os.path.join('wound-segmentation/data/traininig_tfrecord', tfrecord_name)
    count = 0
    writer = tf.io.TFRecordWriter(tf_record_loc)
    img_list = os.listdir(data_loc)
    print ('Writing input data to the {} '.format(tfrecord_name))
    for image in img_list:        
        try:
            img_path = os.path.join(data_loc, image)
            label_path = os.path.join(label_loc, image)
            a = get_image_binary(img_path, label_flag = False)
            b = get_image_binary(label_path, label_flag = True)    
            example = tf.train.Example(features=tf.train.Features(feature =
                                                                  {
                                                                      'image': _bytes_feature(a),
                                                                      'image_name' :  _bytes_feature(bytes(image, 'utf-8')),
                                                                      'image_label': _bytes_feature(b)
                                                                  }))
            writer.write(example.SerializeToString())
        except:
            count += 1
            pass
    writer.close()
    print ('total images skipped during writting tfrecord: {}'.format(count))
    return tf_record_loc

def parse(serialized):
    features = {
                'image' : tf.io.FixedLenFeature([], tf.string),
                'image_name': tf.io.FixedLenFeature([],tf.string),
                'image_label': tf.io.FixedLenFeature([],tf.string)
               }
    parsed_example = tf.io.parse_single_example(serialized, features)
    image_raw = parsed_example['image']
    image_name = parsed_example['image_name']
    image_label = parsed_example['image_label']
    image = tf.io.decode_raw (image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.math.multiply(image, 1./255)
    image = tf.reshape(image, shape=(572,572,1))
    
    image_label = tf.io.decode_raw (image_label, tf.uint8)
    image_label = tf.cast(image_label, tf.float32)
    image_label = tf.reshape(image_label, shape=(388,388,1))
    
    return image, image_name, image_label

def load_tfrecord(predict_file):
    tfrecord_name = predict_file.split('/')[-1]
    print ('Reading "{}" TFRecord for prediction'.format(tfrecord_name))
    dataset = tf.data.TFRecordDataset(filenames = predict_file)
    dataset = dataset.map(parse)
    return (dataset)
    



    