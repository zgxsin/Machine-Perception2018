import time
import os

config = {}

##################################################################
# Please note that the following fields will be updated by us to re-train and re-evaluate your model. 
# Where experiment results are stored.
config['log_dir'] = './runs/'
# In case your pre/post-processing scripts generate intermediate results, you may use config['tmp_dir'] to store them.
config['tmp_dir'] = './tmp/'
# Path to training, validation and test data folders.
# config['train_data_dir'] = "/cluster/work/riner/users/zgxsin/mp2018/dataset/train"
# config['valid_data_dir'] = "/cluster/work/riner/users/zgxsin/mp2018/dataset/validation/"
# config['test_data_dir'] = "/cluster/work/riner/users/zgxsin/mp2018/dataset/test/"

config['train_data_dir'] = "../train/"
config['valid_data_dir'] = "../validation/"
config['test_data_dir'] = "../test/"
##################################################################
# You can modify the rest or add new fields as you need.

# Dataset statistics. You don't need to change unless you use different splits.
config['num_test_samples'] = 2174
config['num_validation_samples'] = 1765
config['num_training_samples'] = 5722

# Hyper-parameters and training configuration.
config['batch_size'] = 16
# config['batch_size'] = 16   ## modified by GX
config['learning_rate'] = 5e-5
# Learning rate is annealed exponentially in 'exponential' case. Don't forget to change annealing configuration in the code.
config['learning_rate_type'] = 'exponential' #'fixed' or 'exponential' or 'decay_by_epochs'
# config['learning_rate_type'] = 'decay_by_epochs'

config['regularization_rate'] = 0.0001 # this is the rate for L2 or L1 regularizer
config['num_steps_per_epoch'] = int(config['num_training_samples']/config['batch_size'])

config['num_epochs'] = 50
# config['num_epochs'] = 50 ## modified by GX
config['evaluate_every_step'] = config['num_steps_per_epoch']*2 # every two epoch, evaluate the model performance
config['checkpoint_every_step'] = config['num_steps_per_epoch']*5 # every 2 epoch save the model
config['num_validation_steps'] = int(config['num_validation_samples']/config['batch_size'])
config['print_every_step'] = 50

# Here I provide three common techniques to calculate sequence loss.
# (1) 'last_logit': calculate loss by using only the last step prediction.
# (2) 'average_logit': calculate loss by using average of predictions across all steps.
# (3) 'average_loss': calculate loss for each time-step by using the same sequence label.
config['loss_type'] = 'average_logit' # 'last_logit', 'average_logit', 'average_loss'.
# config['loss_type'] = 'last_logit' # GX_modified
#config['loss_type'] = 'weighted_logit'

# Dataset and Input Pipeline Configuration
config['inputs'] = {}
config['inputs']['queue_capacity'] = config['batch_size']*20
config['inputs']['num_read_threads'] = 10
config['inputs']['num_epochs'] = config['num_epochs']
config['inputs']['batch_size'] = config['batch_size']
config['inputs']['img_height'] = 80  #Input size.
config['inputs']['img_width'] = 80

# reshape data size:
config['inputs']['img_height_crop'] = 64  #Input size.
config['inputs']['img_width_crop'] = 64
config['inputs']['img_num_channels'] = 3
config['inputs']['skeleton_size'] = 180
## setting the 3dcnn frame lenth and overlap
config['frame_lenth'] = 8
config['real_frame_overlap'] = 2
config['frame_overlap'] = config['frame_lenth'] - config['real_frame_overlap']


# CNN model parameters
config['cnn'] = {}
config['cnn']['num_filters'] = [16,32,64,128,256] # Number of filters for every convolutional layer.
config['cnn']['filter_size'] = [3,3,3,3,3,3] # Kernel size. Assuming kxk kernels.
config['cnn']['num_hidden_units1'] = 1024 # Number of units in the last dense layer, i.e. representation size.
config['cnn']['num_hidden_units2'] = 1024
config['cnn']['dropout_rate'] = 0.25
config['drop_cnn_layers_rate'] = 0.1
# config['cnn']['dropout_rate'] = 0. ## modified by GX
config['cnn']['num_class_labels'] = 20
config['cnn']['batch_size'] = config['batch_size']
config['cnn']['loss_type'] = config['loss_type']
config['cnn']['regularization_rate'] = 0.0001 # this is the rate for L2 or L1 regularizer
config['cnn']['moving_average_decay'] = 0.998 # moving_average_rate
config['cnn']['frame_lenth'] = config['frame_lenth']
config['cnn']['frame_overlap'] = config['frame_overlap']


# RNN model parameters
config['rnn'] = {}
config['rnn']['num_hidden_units'] = 256 # Number of units in an LSTM cell.
# config['rnn']['num_hidden_units'] = 512 # GX add
config['rnn']['dropout_rate'] = 0.4
# config['rnn']['dropout_rate'] = 0.4
config['rnn']['num_layers'] = 1 # Number of LSTM stack.
config['rnn']['num_class_labels'] = 20
config['rnn']['batch_size'] = config['batch_size']
config['rnn']['loss_type'] = config['loss_type']
config['rnn']['regularization_rate'] =  0.0001 # this is the rate for L2 or L1 regularizer
config['rnn']['moving_average_decay'] = 0.998 # moving_average_rate
config['rnn']['frame_lenth'] = config['frame_lenth']
config['rnn']['frame_overlap'] = config['frame_overlap']


# You can set descriptive experiment names or simply set empty string ''.
config['model_name'] = 'lstm1_512_cnn5_drop3_5e4_avg_logit'


# Create a unique output directory for this experiment.
timestamp = str(int(time.time()))
model_folder_name = timestamp if config['model_name'] == '' else config['model_name'] + '_' + timestamp
config['model_id'] = model_folder_name
config['model_dir'] = os.path.abspath(os.path.join(config['log_dir'], model_folder_name))
print("Writing to {}\n".format(config['model_dir']))
