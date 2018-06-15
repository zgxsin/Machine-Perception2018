import time
import os

config = {}

##################################################################
# Please note that the following fields will be updated by us to re-train and re-evaluate your model. 
#
# Where experiment results are stored.
config['log_dir'] = './runs/'
# In case your pre/post-processing scripts generate intermediate results, you may use config['tmp_dir'] to store them.
config['tmp_dir'] = './tmp/'
# Path to training, validation and test data folders.
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
# config['batch_size'] = 10   ## modified by GX
config['learning_rate'] = 5e-4
# Learning rate is annealed exponentially in 'exponential' case. Don't forget to change annealing configuration in the code.
config['learning_rate_type'] = 'exponential' #'fixed' or 'exponential'

config['num_steps_per_epoch'] = int(config['num_training_samples']/config['batch_size'])

config['num_epochs'] = 25
# config['num_epochs'] = 50 ## modified by GX
config['evaluate_every_step'] = config['num_steps_per_epoch']*2
config['checkpoint_every_step'] = config['num_steps_per_epoch']*5
config['num_validation_steps'] = int(config['num_validation_samples']/config['batch_size'])
config['print_every_step'] = 50

# Here I provide three common techniques to calculate sequence loss.
# (1) 'last_logit': calculate loss by using only the last step prediction.
# (2) 'average_logit': calculate loss by using average of predictions across all steps.
# (3) 'average_loss': calculate loss for each time-step by using the same sequence label.
config['loss_type'] = 'average_logit' # 'last_logit', 'average_logit', 'average_loss'.

# Dataset and Input Pipeline Configuration
config['inputs'] = {}
config['inputs']['queue_capacity'] = config['batch_size']*20
config['inputs']['num_read_threads'] = 4
config['inputs']['num_epochs'] = config['num_epochs']
config['inputs']['batch_size'] = config['batch_size']
config['inputs']['img_height'] = 80 # Input size.
config['inputs']['img_width'] = 80
config['inputs']['img_num_channels'] = 3
config['inputs']['skeleton_size'] = 180

# CNN model parameters
config['cnn'] = {}
config['cnn']['num_filters'] = [16,32,64,128,256] # Number of filters for every convolutional layer.
config['cnn']['filter_size'] = [3,3,3,3,3,3] # Kernel size. Assuming kxk kernels.
config['cnn']['num_hidden_units'] = 512 # Number of units in the last dense layer, i.e. representation size.
# config['cnn']['dropout_rate'] = 0.3
config['cnn']['dropout_rate'] = 0.4 ## modified by GX
config['cnn']['num_class_labels'] = 20
config['cnn']['batch_size'] = config['batch_size']
config['cnn']['loss_type'] = config['loss_type']

# RNN model parameters
config['rnn'] = {}
config['rnn']['num_hidden_units'] = 512 # Number of units in an LSTM cell.
config['rnn']['dropout_rate'] = 0.3
config['rnn']['num_layers'] = 1 # Number of LSTM stack.
config['rnn']['num_class_labels'] = 20
config['rnn']['batch_size'] = config['batch_size']
config['rnn']['loss_type'] = config['loss_type']


# You can set descriptive experiment names or simply set empty string ''.
config['model_name'] = 'lstm1_512_cnn5_drop3_5e4_avg_logit'


# Create a unique output directory for this experiment.
timestamp = str(int(time.time()))
model_folder_name = timestamp if config['model_name'] == '' else config['model_name'] + '_' + timestamp
config['model_id'] = model_folder_name
config['model_dir'] = os.path.abspath(os.path.join(config['log_dir'], model_folder_name))
print("Writing to {}\n".format(config['model_dir']))
