import os
import time
import json

import tensorflow as tf
from model_input import input_pipeline
from model import CNNModel, RNNModel

from Skeleton import Skeleton





def main(config):
    # TODO
    # Here you can call your preprocessing functions. If you generate intermediate representations, you should be
    # using config['tmp_dir'] directory.
    # If you use a different training/validation split than what we provide, please make sure that this split is
    # reproducible. You can either set `seed` or save the split indices into a  file and submit it along with your code.

    #############
    # Data
    #############

    # Each <key,value> pair in `training_placeholders` and `validation_placeholders` corresponds to Tensorflow placeholder.
    # Alternatively we could load data into memory and feed to the model by using feed_dict approach.
    # Create input placeholders for training data.
    train_tfrecord_files = [os.path.join(config['train_data_dir'], "dataTrain_%d.tfrecords"%i) for i in range(1, 41)]
    training_placeholders = input_pipeline(tfrecord_files=train_tfrecord_files,
                                           config=config['inputs'],
                                           name='training_input_pipeline',
                                           shuffle=True)

    # Create input placeholders for validation data.
    valid_tfrecord_files = [os.path.join(config['valid_data_dir'], "dataValidation_%d.tfrecords"%i) for i in range(1, 16)]
    validation_placeholders = input_pipeline(tfrecord_files=valid_tfrecord_files,
                                               config=config['inputs'],
                                               name='validation_input_pipeline',
                                               shuffle=False)
    ###############




    # add normalized depth info to the CNN training data, replace rgb with mask_image
    training_input_layer = tf.concat([training_placeholders['mask'],  training_placeholders['skeleton'],training_placeholders['depth']],4)
    validation_input_layer = tf.concat([validation_placeholders['mask'], validation_placeholders['skeleton'],validation_placeholders['depth']], 4 )

    ##################
    # Training Model
    ##################
    # Create separate graphs for training and validation.
    # Training graph.
    global_step = tf.Variable(1, name='global_step', trainable=False )
    # apply moving average
    ema = tf.train.ExponentialMovingAverage(0.998, global_step)
    with tf.name_scope("Training"):
        # Create model
        cnnModel = CNNModel(config=config['cnn'],
                            placeholders=training_placeholders,
                            mode='training',ema= ema)
        cnnModel.build_graph(input_layer=training_input_layer)

        trainModel = RNNModel(config=config['rnn'],
                              placeholders=training_placeholders,
                              mode="training", ema = ema)
        ### add training skeleton info to the input layer of RNN
        # input2rnn = tf.concat([training_placeholders['skeleton'], cnnModel.model_output], 2)
        trainModel.build_graph(input_layer=cnnModel.model_output)
        trainModel.build_loss()
        # apply moving average
        ema_op = ema.apply(tf.trainable_variables())
        print("\n# of parameters: %s"%trainModel.get_num_parameters())

        ##############
        # Optimization
        ##############
        if config['learning_rate_type'] == 'exponential':
            learning_rate = tf.train.exponential_decay(config['learning_rate'],
                                                       global_step=global_step,
                                                       decay_steps=500,
                                                       decay_rate=0.97,
                                                       staircase=False)
        elif config['learning_rate_type'] == 'fixed':
            learning_rate = config['learning_rate']
        else:
            raise Exception("Invalid learning rate type")

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op1 = optimizer.minimize(trainModel.loss, global_step=global_step)


        with tf.control_dependencies([train_op1, ema_op]):
            train_op = tf.no_op(name="train_ema")
    ###################
    # Validation Model
    ###################
    with tf.name_scope("Validation"):
        # Create model
        validCnnModel = CNNModel(config=config['cnn'],
                            placeholders=validation_placeholders,
                            mode='validation', ema=ema)
        validCnnModel.build_graph(input_layer=validation_input_layer)

        validModel = RNNModel(config=config['rnn'],
                              placeholders=validation_placeholders,
                              mode="validation", ema=ema)
        ### add training skeleton info to the input layer of RNN
        # input2rnn_val = tf.concat([validation_placeholders['skeleton'], validCnnModel.model_output],2)
        validModel.build_graph(input_layer=validCnnModel.model_output)
        validModel.build_loss()

    ##############
    # Monitoring
    ##############
    # Create placeholders to provide tensorflow average loss and accuracy.
    loss_avg_pl = tf.placeholder(tf.float32, name="loss_avg_pl")
    accuracy_avg_pl = tf.placeholder(tf.float32, name="accuracy_avg_pl")

    # Create summary ops for monitoring the training.
    # Each summary op annotates a node in the computational graph and plots evaluation results.
    summary_train_loss = tf.summary.scalar('loss', trainModel.loss)
    summary_train_acc = tf.summary.scalar('accuracy_training', trainModel.batch_accuracy)
    summary_avg_accuracy = tf.summary.scalar('accuracy_avg', accuracy_avg_pl)
    summary_avg_loss = tf.summary.scalar('loss_avg', loss_avg_pl)
    summary_learning_rate = tf.summary.scalar('learning_rate', learning_rate)

    # Group summaries. summaries_training is used during training and reported after every step.
    summaries_training = tf.summary.merge([summary_train_loss, summary_train_acc, summary_learning_rate])
    # summaries_evaluation is used by both training and validation in order to report the performance on the dataset.
    summaries_evaluation = tf.summary.merge([summary_avg_accuracy, summary_avg_loss])

    # Create session object
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))


    ##############################
    # Restoring and Initialization  DaiQi_add
    ##############################

    # Load Previous Model and initialize weights
    # restored_variables = tf.trainable_variables()
    # restore_saver = tf.train.Saver(var_list=restored_variables )
    # # latest_checkpoint(checkpoint_dir, latest_filename=None)
    # # checkpoint_path = tf.train.latest_checkpoint("~/Workspace/Data/zgxsin_data/source_code/runs/lstm1_512_cnn5_drop3_5e4_avg_logit_1526236758 ")
    # checkpoint_path = tf.train.latest_checkpoint(
    #     " /Users/zhou/MP2018_Edit/mp18-dynamic-gesture-recognition/source_code/runs/lstm1_512_cnn5_drop3_5e4_avg_logit_1526652032")
    # print('Restoring from ', checkpoint_path)
    # restore_saver.restore(session, checkpoint_path )
    #
    # # Initialize remaining uninitialized variables
    # all_variables = tf.global_variables() + tf.local_variables()
    # initialized_list = []
    # for varIdx in range(len(all_variables)):
    #     variable = all_variables[varIdx]
    #     varFlag = session.run(tf.is_variable_initialized(variable))
    #     if not varFlag:
    #         initialized_list.append(variable )
    #
    # init_op = tf.variables_initializer(initialized_list, name='init_remaining' )
    # session.run(init_op)

    ##############################
    # Restoring and Initialization  DaiQi_add
    ##############################


    # Add the ops to initialize variables.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # Actually initialize the variables
    session.run(init_op)

    # Register summary ops.
    train_summary_dir = os.path.join(config['model_dir'], "summary", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)
    valid_summary_dir = os.path.join(config['model_dir'], "summary", "valid")
    valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, session.graph)

    # Create a saver for saving checkpoints.
    # save EMA variables and other variables
    restore_variables = tf.trainable_variables() \
                        + tf.moving_average_variables()
    # saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=3, save_relative_paths=True)
    # save all the variables
    saver = tf.train.Saver(max_to_keep=3, save_relative_paths=True )
    # Define counters in order to accumulate measurements.
    counter_correct_predictions_training = 0.0
    counter_loss_training = 0.0
    counter_correct_predictions_validation = 0.0
    counter_loss_validation = 0.0

    # Save configuration in json formats.
    json.dump(config, open(os.path.join(config['model_dir'], 'config.json'), 'w'), indent=4, sort_keys=True)

    ##########################
    # Training Loop
    ##########################
    # Initialize data I/O threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)
    step = 0
    try:
        while not coord.should_stop():
            step = tf.train.global_step(session, global_step)

            if (step%config['checkpoint_every_step']) == 0:
                ckpt_save_path = saver.save(session, os.path.join(config['model_dir'], 'model'), global_step)
                print("Model saved in file: %s"%ckpt_save_path)

            start_time = time.perf_counter()
            # Run the optimizer to update weights.
            # Note that "train_op" is responsible from updating network weights.
            # Only the operations that are fed are evaluated.
            # Run the optimizer to update weights.
            train_summary, num_correct_predictions, loss, _ = session.run([summaries_training,
                                                                        trainModel.num_correct_predictions,
                                                                        trainModel.loss,
                                                                        train_op],
                                                                       feed_dict={})
            # visual_skele = session.run([training_placeholders['skeleton']])
            # visual_rgb = session.run( [training_placeholders['rgb']] )
            # import matplotlib.pyplot as plt
            # plt.imshow(visual_skele[0][0])
            # plt.show()

            # Update counters.
            counter_correct_predictions_training += num_correct_predictions
            counter_loss_training += loss
            # Write summary data.
            train_summary_writer.add_summary(train_summary, step)

            # Report training performance
            if (step%config['print_every_step']) == 0:
                # To get a smoother loss plot, we calculate average performance.
                accuracy_avg = counter_correct_predictions_training/(config['batch_size']*config['print_every_step'])
                loss_avg = counter_loss_training/(config['print_every_step'])
                # Feed average performance.
                summary_report = session.run(summaries_evaluation,
                                             feed_dict={accuracy_avg_pl: accuracy_avg, loss_avg_pl: loss_avg})
                train_summary_writer.add_summary(summary_report, step)
                time_elapsed = (time.perf_counter() - start_time)/config['print_every_step']
                print("[Train/%d] Accuracy: %.3f, Loss: %.3f, time/step = %.3f"%(step,
                                                                                 accuracy_avg,
                                                                                 loss_avg,
                                                                                 time_elapsed))
                counter_correct_predictions_training = 0.0
                counter_loss_training = 0.0

            # Report validation performance
            if (step%config['evaluate_every_step']) == 0:
                # We create a input queue for validation data for multiple epochs.
                # Note that we approximate one validation epoch (validation doesn't have to be accurate.)
                # In other words, number of unique validation samples the model sees may differ every time.
                start_time = time.perf_counter()
                for eval_step in range(config['num_validation_steps']):
                    # Calculate average validation accuracy.
                    num_correct_predictions, loss = session.run([validModel.num_correct_predictions,
                                                                 validModel.loss])
                    # Update counters.
                    counter_correct_predictions_validation += num_correct_predictions
                    counter_loss_validation += loss

                # Report validation performance
                accuracy_avg = counter_correct_predictions_validation/(config['batch_size']*config['num_validation_steps'])
                loss_avg = counter_loss_validation/(config['num_validation_steps'])
                summary_report = session.run(summaries_evaluation,
                                          feed_dict={accuracy_avg_pl: accuracy_avg, loss_avg_pl: loss_avg})
                valid_summary_writer.add_summary(summary_report, step)
                time_elapsed = (time.perf_counter() - start_time)/config['num_validation_steps']
                print("[Valid/%d] Accuracy: %.3f, Loss: %.3f, time/step = %.3f"%(step,
                                                                                      accuracy_avg,
                                                                                      loss_avg,
                                                                                      time_elapsed))

                counter_correct_predictions_validation = 0.0
                counter_loss_validation = 0.0

    except tf.errors.OutOfRangeError:
        print('Model is trained for %d epochs, %d steps.'%(config['num_epochs'], step))
        print('Done.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    ckpt_save_path = saver.save(session, os.path.join(config['model_dir'], 'model'), global_step)
    print("Model saved in file: %s"%ckpt_save_path)
    session.close()

    # Evaluate model after training and create submission file.
    tf.reset_default_graph()
    from restore_and_evaluate import main as evaluate
    config['checkpoint_id'] = None
    evaluate(config)


if __name__ == '__main__':
    from config import config
    main(config)