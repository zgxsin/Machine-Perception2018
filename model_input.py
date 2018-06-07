import tensorflow as tf
from Skeleton import Skeleton
import numpy as np


def get_mean_and_std(tensor, axis, keepdims=False):
    """
    Calculates mean and standard deviation of a tensor over given dimensions.
    :param tensor:
    :param axis:
    :param keepdims:
    :return:
    """
    mean = tf.reduce_mean(tensor, axis=axis, keepdims=True)
    diff_squared = tf.square(tensor - mean)
    variance = tf.reduce_mean(diff_squared, axis=axis, keepdims=keepdims)
    std = tf.maximum(tf.sqrt(variance), 1e-6)

    return mean, std


def applyMask(img, segmentedUser):
    """
    Applies mask on the given image for visualization.
    """
    if len(img.shape) > 2: # Color image
        mask3 = segmentedUser > 150
        masked_img = img * mask3
    else:
        mask2 = np.mean(segmentedUser, axis=2) > 150
        masked_img = img * mask2
    return masked_img


def img_preprocessing_op(image_op):
    """
    Creates preprocessing operations that are going to be applied on a single frame.

    You can do any preprocessing (masking, normalization/scaling of inputs, augmentation, etc.) by using tensorflow
    operations. Here I provided some examples commented in the code. You can find more built-in image operations at
    https://www.tensorflow.org/api_docs/python/tf/image .

    :param image_op:
    :return:
    """
    with tf.name_scope("img_preprocessing"):
        # Convert from RGB to greyscale.
        # image_op = tf.image.rgb_to_grayscale(image_op)

        # Crop
        #image_op = tf.image.resize_image_with_crop_or_pad(image_op, 60, 60)

        # Resize operation requires 4D tensors (i.e., batch of images).
        # Reshape the image so that it looks like a batch of one sample: [1,60,60,1]
        #image_op = tf.expand_dims(image_op, 0)
        # Resize
        #image_op = tf.image.resize_bilinear(image_op, np.asarray([32,32]))
        # Reshape the image: [32,32,1]
        #image_op = tf.squeeze(image_op, 0)

        # Normalize (zero-mean unit-variance) the image locally, i.e., by using statistics of the
        # image not the whole data or sequence.
        # image_op = tf.image.per_image_standardization(image_op)

        # Flatten image
        # y.set_shape(image_op.get_shape())
        # image_op_ = tf.TensorArray(
        #     dtype=tf.float32, size=0, dynamic_size=True)

        image_op_ = np.ones(shape=[image_op.shape[0], 80, 80, 3], dtype=np.float32)

        for i in range(image_op.shape[0]):
            skeleton = Skeleton(image_op[i])
            skeleton.resizePixelCoordinates()
            image_op_[i] = skeleton.toImage(80, 80)
        # return skeletonimage [0,255]
        return image_op_.astype(np.float32)

def img_preprocessing_op_map(image_op):
    """
    Creates preprocessing operations that are going to be applied on a single frame.

    You can do any preprocessing (masking, normalization/scaling of inputs, augmentation, etc.) by using tensorflow
    operations. Here I provided some examples commented in the code. You can find more built-in image operations at
    https://www.tensorflow.org/api_docs/python/tf/image .

    :param image_op:
    :return:
    """
    with tf.name_scope("img_preprocessing"):
        # Convert from RGB to greyscale.
        # image_op = tf.image.rgb_to_grayscale(image_op)

        # Crop
        #image_op = tf.image.resize_image_with_crop_or_pad(image_op, 60, 60)

        # Resize operation requires 4D tensors (i.e., batch of images).
        # Reshape the image so that it looks like a batch of one sample: [1,60,60,1]
        #image_op = tf.expand_dims(image_op, 0)
        # Resize
        #image_op = tf.image.resize_bilinear(image_op, np.asarray([32,32]))
        # Reshape the image: [32,32,1]
        #image_op = tf.squeeze(image_op, 0)

        # Normalize (zero-mean unit-variance) the image locally, i.e., by using statistics of the
        # image not the whole data or sequence.
        # image_op = tf.image.per_image_standardization(image_op)

        # Flatten image
        # y.set_shape(image_op.get_shape())
        # image_op_ = tf.TensorArray(
        #     dtype=tf.float32, size=0, dynamic_size=True)
        # image_op = tf.image.resize_image_with_crop_or_pad(image_op, target_height=0.7*image_op.shape[0].value,
        #                                        target_width= 0.5*image_op.shape[1].value)

        image_op = tf.image.resize_image_with_crop_or_pad(image_op, target_height=80,
                                                           target_width=80)

        image_op_post = tf.image.random_flip_left_right(image_op, seed=6)


        return image_op_post


def read_and_decode_sequence(filename_queue, config):
    # Create a TFRecordReader.
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions)
    _, serialized_example = reader.read(filename_queue)

    # Read one sequence sample.
    # The training and validation files contains the following fields:
    # - label: label of the sequence which take values between 1 and 20.
    # - length: length of the sequence, i.e., number of frames.
    # - depth: sequence of depth images. [length x height x width x 1]
    # - rgb: sequence of rgb images. [length x height x width x 3]
    # - segmentation: sequence of segmentation masks. [length x height x width x num_channels]
    # - skeleton: sequence of flattened skeleton joint positions. [length x num_joints]
    #
    # The test files doesn't contain "label" field.
    # [height, width, num_channels] = [80, 80, 3]
    with tf.name_scope("TFRecordDecoding"):
        context_encoded, sequence_encoded = tf.parse_single_sequence_example(
                serialized_example,
                # "label" and "length" are encoded as context features.
                context_features={
                    "label": tf.FixedLenFeature([], dtype=tf.int64),
                    "length": tf.FixedLenFeature([], dtype=tf.int64)
                },
                # "depth", "rgb", "segmentation", "skeleton" are encoded as sequence features.
                sequence_features={
                    "depth": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "segmentation": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "skeleton": tf.FixedLenSequenceFeature([], dtype=tf.string),
                })

        # Fetch and decode the serialized data.
        seq_rgb = tf.decode_raw(sequence_encoded['rgb'], tf.uint8)
        seq_depth = tf.decode_raw(sequence_encoded['depth'], tf.uint8)
        seq_segmentation = tf.decode_raw(sequence_encoded['segmentation'], tf.uint8)
        seq_skeleton = tf.decode_raw(sequence_encoded['skeleton'], tf.float32)

        seq_label = context_encoded['label']
        # Tensorflow requires the labels start from 0. Before you create submission csv,
        # increment the predictions by 1.
        seq_label = seq_label - 1
        seq_len = tf.to_int32(context_encoded['length'])

        # Reshape data modalities.
        seq_rgb = tf.to_float(tf.reshape(seq_rgb, (-1, config['img_height'], config['img_width'], 3)))
        seq_depth = tf.to_float(tf.reshape(seq_depth, (-1, config['img_height'], config['img_width'], 1)))
        seq_segmentation = tf.to_float(tf.reshape(seq_segmentation, (-1, config['img_height'], config['img_width'], 3)))
        seq_skeleton = tf.to_float(tf.reshape(seq_skeleton, (-1, 180)))

        ###############################
        # TODO
        # Here you can apply preprocessing/augmentation on input frames (it is commented).
        # tf.map_fn applies the preprocessing function on every image in the sequence, i.e., frame.

        # skeleton_dim = 80*80*3*tf.shape(seq_skeleton)[0]
        # [tf.float32]*skeleton_dim.

        # seq_skeleton [0,255]
        seq_skeleton = tf.py_func( lambda x: img_preprocessing_op( x ),
                                   [seq_skeleton],
                                   tf.float32,
                                   )
        seq_skeleton.set_shape( [None, 80, 80, 3] )

        mask_result = tf.py_func( lambda x: x > 150,
                                  [seq_segmentation],
                                  tf.bool,
                                  )
        mask_result.set_shape( [None, 80, 80, 3] )
        # get the human shape mask image, normalized
        image_extracted = tf.cast( mask_result, tf.float32 ) * seq_rgb / 255

        # image_extracted = tf.map_fn(lambda x: tf.image.random_flip_left_right(x, seed=6),
        #                              elems=image_extracted,
        #                              dtype=tf.float32,
        #                              back_prop=False
        #                              )




        # Normalize RGB images before feeding into the model.
        # TODO
        # Here we calculate statistics locally (i.e., per sequence sample). You can iterate over the whole dataset once
        # and calculate global statistics for later use.
        rgb_mean, rgb_std = get_mean_and_std( seq_rgb, axis=[0, 1, 2, 3], keepdims=True )
        seq_rgb = (seq_rgb - rgb_mean) / rgb_std

        skeleton_mean, skeleton_std = get_mean_and_std( seq_skeleton, axis=[0, 1, 2, 3], keepdims=True )
        seq_skeleton = (seq_skeleton - skeleton_mean) / skeleton_std
        # normalize tp [-1,1]
        seq_depth = seq_depth / 255
        # Create a dictionary containing a sequence sample in different modalities. Tensorflow creates mini-batches in
        # the same format.
        sample = {}
        sample['rgb'] = seq_rgb
        sample['depth'] = seq_depth
        sample['segmentation'] = seq_segmentation
        sample['skeleton'] = seq_skeleton
        sample['seq_len'] = seq_len
        sample['labels'] = seq_label
        sample['mask'] = image_extracted

        return sample


def read_and_decode_sequence_test_data(filename_queue, config):
    """
    Replace label field with id field because test data doesn't contain labels.
    """
    # Create a TFRecordReader.
    readerOptions = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    reader = tf.TFRecordReader(options=readerOptions)
    _, serialized_example = reader.read(filename_queue)

    # Read one sequence sample.
    # The training and validation files contains the following fields:
    # - id: id of the sequence samples which is used to create submission file.
    # - length: length of the sequence, i.e., number of frames.
    # - depth: sequence of depth images. [length x height x width x numChannels]
    # - rgb: sequence of rgb images. [length x height x width x numChannels]
    # - segmentation: sequence of segmentation masks. [length x height x width x numChannels]
    # - skeleton: sequence of flattened skeleton joint positions. [length x numJoints]
    #
    # The test files doesn't contain "label" field.
    # [height, width, numChannels] = [80, 80, 3]
    with tf.name_scope("TFRecordDecoding"):
        context_encoded, sequence_encoded = tf.parse_single_sequence_example(
                serialized_example,
                # "label" and "length" are encoded as context features.
                context_features={
                    "id": tf.FixedLenFeature([], dtype=tf.int64),
                    "length": tf.FixedLenFeature([], dtype=tf.int64)
                },
                # "depth", "rgb", "segmentation", "skeleton" are encoded as sequence features.
                sequence_features={
                    "depth": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "rgb": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "segmentation": tf.FixedLenSequenceFeature([], dtype=tf.string),
                    "skeleton": tf.FixedLenSequenceFeature([], dtype=tf.string),
                })

        # Fetch and decode the serialized data.
        seq_rgb = tf.decode_raw(sequence_encoded['rgb'], tf.uint8)
        seq_depth = tf.decode_raw(sequence_encoded['depth'], tf.uint8)
        seq_segmentation = tf.decode_raw(sequence_encoded['segmentation'], tf.uint8)
        seq_skeleton = tf.decode_raw(sequence_encoded['skeleton'], tf.float32)

        seq_id = context_encoded['id']
        seq_len = tf.to_int32(context_encoded['length'])

        # Reshape data modalities.
        seq_rgb = tf.to_float(tf.reshape(seq_rgb, (-1, config['img_height'], config['img_width'], 3)))
        seq_depth = tf.to_float(tf.reshape(seq_depth, (-1, config['img_height'], config['img_width'], 1)))
        seq_segmentation = tf.to_float(tf.reshape(seq_segmentation, (-1, config['img_height'], config['img_width'], 3)))
        seq_skeleton = tf.reshape(seq_skeleton, (seq_len, 180))

        ###############################
        # TODO
        # Here you can apply preprocessing/augmentation on input frames (it is commented).
        # tf.map_fn applies the preprocessing function on every image in the sequence, i.e., frame.
        # seq_rgb = tf.map_fn(lambda x: img_preprocessing_op(x),
        #                    elems=seq_rgb,
        #                    dtype=tf.float32,
        #                    back_prop=False)
        seq_skeleton = tf.py_func( lambda x: img_preprocessing_op( x ),
                                   [seq_skeleton],
                                   tf.float32,
                                   )
        seq_skeleton.set_shape( [None, 80, 80, 3] )

        mask_result = tf.py_func( lambda x: x > 150,
                                  [seq_segmentation],
                                  tf.bool,
                                  )
        mask_result.set_shape( [None, 80, 80, 3] )
        # get the human shape mask image, normalized
        image_extracted = tf.cast( mask_result, tf.float32 ) * seq_rgb / 255

        # image_extracted = tf.map_fn(lambda x: tf.image.random_flip_left_right(x, seed=6),
        #                              elems=image_extracted,
        #                              dtype=tf.float32,
        #                              back_prop=False
        #                              )




        # Normalize RGB images before feeding into the model.
        # TODO
        # Here we calculate statistics locally (i.e., per sequence sample). You can iterate over the whole dataset once
        # and calculate global statistics for later use.
        rgb_mean, rgb_std = get_mean_and_std( seq_rgb, axis=[0, 1, 2, 3], keepdims=True )
        seq_rgb = (seq_rgb - rgb_mean) / rgb_std

        skeleton_mean, skeleton_std = get_mean_and_std( seq_skeleton, axis=[0, 1, 2, 3], keepdims=True )
        seq_skeleton = (seq_skeleton - skeleton_mean) / skeleton_std
        # no flips
        # seq_skeleton = tf.map_fn(lambda x: tf.image.random_flip_left_right(x, seed=6),
        #                              elems=seq_skeleton,
        #                              dtype=tf.float32,
        #                              back_prop=False
        #                              )

        seq_depth = seq_depth / 255

        # Create a dictionary containing a sequence sample in different modalities. Tensorflow creates mini-batches in
        # the same format.
        sample = {}
        sample['rgb'] = seq_rgb
        sample['depth'] = seq_depth
        sample['segmentation'] = seq_segmentation
        sample['skeleton'] = seq_skeleton
        sample['seq_len'] = seq_len
        sample['ids'] = seq_id
        sample['mask'] = image_extracted
        return sample


def input_pipeline(tfrecord_files, config, name='input_pipeline', shuffle=True, mode='training'):
    """
    Creates Tensorflow input pipeline. Multiple threads read, decode, preprocess and enqueue data samples. Mini-batches
    of padded variable-length sequences are generated for model.

    :param tfrecord_files: list of tfrecord data file paths.
    :param config: configuration of input I/O.
    :param name:
    :param shuffle:
    :param mode:
    :return:
    """
    with tf.name_scope(name):
        # Read the data from TFRecord files, decode and create a list of data samples by using multiple threads.
        if mode is "training":
            # Create a queue of TFRecord input files.
            filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=config['num_epochs'], shuffle=shuffle)
            sample_list = [read_and_decode_sequence(filename_queue, config) for _ in range(config['num_read_threads'])]
            batch_sample = tf.train.batch_join(sample_list,
                                               batch_size=config['batch_size'],
                                               capacity=config['queue_capacity'],
                                               enqueue_many=False,
                                               dynamic_pad=True,
                                               allow_smaller_final_batch=False,
                                               name="batch_join_and_pad")
            return batch_sample

        else:
            filename_queue = tf.train.string_input_producer(tfrecord_files, num_epochs=1, shuffle=False)
            sample_list = [read_and_decode_sequence_test_data(filename_queue, config)]
            batch_sample = tf.train.batch_join(sample_list,
                                               batch_size=config['batch_size'],
                                               capacity=config['queue_capacity'],
                                               enqueue_many=False,
                                               dynamic_pad=True,
                                               allow_smaller_final_batch=False,
                                               name="batch_join_and_pad")
            return batch_sample