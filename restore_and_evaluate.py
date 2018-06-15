import tensorflow as tf
import os
import argparse
import json

from model_input import  input_pipeline
from model import CNNModel, RNNModel
from utils import createSubmissionFile

def main(config):
    config['batch_size'] = 2 # Divisor of number of test samples. Don't change it.
    config['rnn']['batch_size'] = 2
    config['cnn']['batch_size'] = 2
    config['inputs']['batch_size'] = 2

    # Create input placeholders for test data.
    test_tfrecord_files = [os.path.join(config['test_data_dir'], "dataTest_%d.tfrecords"%i) for i in range(1, 16)]
    test_placeholders = input_pipeline(tfrecord_files=test_tfrecord_files,
                                       config=config['inputs'],
                                       name='test_input_pipeline',
                                       shuffle=False,
                                       mode="inference")

    test_input_layer = test_placeholders['rgb']

    session = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    # Test graph.
    with tf.name_scope("Inference"):
        # Create model
        inferCnnModel = CNNModel(config=config['cnn'],
                                 placeholders=test_placeholders,
                                 mode='inference')
        inferCnnModel.build_graph(input_layer=test_input_layer)

        inferModel = RNNModel(config=config['rnn'],
                              placeholders=test_placeholders,
                              mode="inference")
        inferModel.build_graph(input_layer=inferCnnModel.model_output)
        inferModel.build_loss()

    # Restore computation graph.
    saver = tf.train.Saver(save_relative_paths=True)
    # Restore variables.
    checkpoint_path = config['checkpoint_id']
    if checkpoint_path is None:
        checkpoint_path = tf.train.latest_checkpoint(config['model_dir'])
    else:
        pass
    print("Evaluating " + checkpoint_path)
    saver.restore(session, checkpoint_path)

    # Evaluation loop
    test_predictions = []
    test_sample_ids = []
    try:
        while not coord.should_stop():
            # Get predicted labels and sample ids for submission csv.
            [predictions, sample_ids] = session.run([inferModel.predictions, test_placeholders['ids']], feed_dict={})
            test_predictions.extend(predictions)
            test_sample_ids.extend(sample_ids)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    # Writes submission file.
    sorted_labels = [label for _, label in sorted(zip(test_sample_ids, test_predictions))]
    createSubmissionFile(sorted_labels, outputFile=os.path.join(config['model_dir'], 'submission_' + config['model_id'] + '.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--model_save_dir', dest='log_dir', type=str, default='./runs/', help='path to main model save directory')
    parser.add_argument('-M', '--model_name', dest='model_name', type=str, help='model folder')
    parser.add_argument('-C', '--checkpoint_id', type=str, default=None, help='checkpoint id (only step number)')
    args = parser.parse_args()

    experiment_dir = os.path.abspath(os.path.join(args.log_dir, args.model_name))
    # Loads config file from experiment folder.
    config = json.load(open(os.path.abspath(os.path.join(args.log_dir, args.model_name, 'config.json')), 'r'))
    if args.checkpoint_id is not None:
        config['checkpoint_id'] = os.path.join(experiment_dir, 'model-' + str(args.checkpoint_id))
    else:
        config['checkpoint_id'] = None # The latest checkpoint will be used.

    main(config)