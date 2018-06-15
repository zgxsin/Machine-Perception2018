import tensorflow as tf

# Note that tf.variable_scope enables sharing the parameters so that both training and validation models share the
# same parameters.

class Model():
    """
    Base class for sequence models.
    """
    def __init__(self, config, placeholders, mode):
        """
        :param config: dictionary of hyper-parameters.
        :param placeholders: dictionary of input placeholders so that you can pass different modalities.
        :param mode: running mode.
        """

        self.config = config
        self.input_placeholders = placeholders

        assert mode in ["training", "validation", "inference"]
        self.mode = mode
        self.is_training = self.mode == "training"
        self.reuse = self.mode == "validation"

        self.input_seq_len = placeholders['seq_len']
        if self.mode is not "inference":
            self.input_target_labels = placeholders['labels']

        self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.input_seq_len, dtype=tf.float32), -1)

        # Total number of trainable parameters.
        self.num_parameters = 0

        # Set by build_graph method.
        self.input_layer = None

        # This member variable is assumed to be set by build_network() method. It is final layer.
        self.model_output_raw = None

        # These member variables are assumed to be set by build_graph() method.
        # Model outputs with shape [batch_size, seq_len, representation_size]
        self.model_output = None
        # Model outputs with shape [batch_size*seq_len, representation_size]
        self.model_output_flat = None

        #self.initializer = tf.contrib.layers.xavier_initializer()
        self.initializer = tf.glorot_normal_initializer()

    def build_graph(self, input_layer=None):
        """
        Called externally. Builds tensorflow graph by calling build_network. Applies preprocessing on the inputs and
        postprocessing on model outputs.

        :param input_layer: External input. Provides an interface for stacking arbitrary models. For example, RNN model
                            can be fed with output representation of a CNN model.
        """
        raise NotImplementedError('subclasses must override build_graph()')

    def build_network(self):
        """
        Builds internal dynamics of the model. Sets
        """
        raise NotImplementedError('subclasses must override build_network()')

    def build_loss(self):
        """
        Calculates classification loss depending on loss type. We are trying to assign a class label to input
        sequences (i.e., many to one mapping). We need to reduce sequence information into a single step by either
        selecting the last step or taking average over all steps. You are welcome to implement a more sophisticated
        approach.
        """
        # Calculate logits
        with tf.variable_scope('logits', reuse=self.reuse, initializer=self.initializer, regularizer=None):
            dropout_layer = tf.layers.dropout(inputs=self.model_output_flat, rate=self.config['dropout_rate'], training=self.is_training)
            logits_non_temporal = tf.layers.dense(inputs=dropout_layer, units=self.config['num_class_labels'])
            self.logits = tf.reshape(logits_non_temporal, [self.config['batch_size'], -1, self.config['num_class_labels']])


        with tf.variable_scope('logits_prediction', reuse=self.reuse, initializer=self.initializer, regularizer=None):
            if self.config['loss_type'] == 'last_logit': # Select the last step. Note that we have variable-length and padded sequences.
                self.logits = tf.gather_nd(self.logits, tf.stack([tf.range(self.config['batch_size']), self.input_seq_len - 1], axis=1))
                self.accuracy_logit = self.logits
            elif self.config['loss_type'] == 'average_logit': # Take average of time steps.
                self.logits = tf.reduce_mean(self.logits*self.seq_loss_mask, axis=1)
                self.accuracy_logit = self.logits
            elif self.config['loss_type'] == 'average_loss':
                self.accuracy_logit = tf.reduce_mean(self.logits*self.seq_loss_mask, axis=1)
            else:
                raise Exception("Invalid loss type")

        if self.mode is not "inference":
            with tf.name_scope("cross_entropy_loss"):
                if self.config['loss_type'] == 'average_loss':
                    labels_all_steps = tf.tile(tf.expand_dims(self.input_target_labels, dim=1), [1, tf.reduce_max(self.input_seq_len)])
                    self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                                 targets=labels_all_steps,
                                                                 weights=self.seq_loss_mask[:, :, 0],
                                                                 average_across_timesteps=True,
                                                                 average_across_batch=True)
                else:
                    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_target_labels))

            with tf.name_scope("accuracy_stats"):
                # Return a bool tensor with shape [batch_size] that is true for the correct predictions.
                self.correct_predictions = tf.equal(tf.argmax(self.accuracy_logit, 1), self.input_target_labels)
                # Number of correct predictions in order to calculate average accuracy afterwards.
                self.num_correct_predictions = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32))
                # Calculate the accuracy per mini-batch.
                self.batch_accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

        # Accuracy calculation.
        with tf.name_scope("accuracy"):
            # Return list of predictions (useful for making a submission)
            self.predictions = tf.argmax(self.accuracy_logit, 1, name="predictions")


    def get_num_parameters(self):
        """
        :return: total number of trainable parameters.
        """
        # Iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # multiplying dimension values
            self.num_parameters += local_parameters

        return self.num_parameters


class CNNModel(Model):
    """
    Convolutional neural network for sequence modeling.
    - Accepts inputs of rank 5 where a mini-batch has shape of [batch_size, seq_len, height, width, num_channels].
    - Ignores temporal dependency.
    """
    def __init__(self, config, placeholders, mode):
        super().__init__(config, placeholders, mode)

        self.input_rgb = placeholders['rgb']

    def build_network(self):
        """
        Stacks convolutional layers where each layer consists of CNN+Pooling operations.
        """
        with tf.variable_scope("convolution", reuse=self.reuse, initializer=self.initializer, regularizer=None):
            input_layer_ = self.input_layer
            for i, num_filter in enumerate(self.config['num_filters']):
                conv_layer = tf.layers.conv2d(inputs=input_layer_,
                                              filters=num_filter,
                                              kernel_size=[self.config['filter_size'][i], self.config['filter_size'][i]],
                                              padding="same",
                                              activation=tf.nn.relu)

                pooling_layer = tf.layers.max_pooling2d(inputs=conv_layer, pool_size=[2, 2], strides=2, padding='same')
                input_layer_ = pooling_layer

            self.model_output_raw = input_layer_

    def build_graph(self, input_layer=None):
        with tf.variable_scope("cnn_model", reuse=self.reuse, initializer=self.initializer, regularizer=None):
            if input_layer is None:
                # Here we use RGB modality only.
                self.input_layer = self.input_rgb
            else:
                self.input_layer = input_layer

            # Input of convolutional layers must have shape [batch_size, height, width, num_channels].
            # Since convolution operation doesn't utilize temporal information, we reshape input sequences such that each
            # frame is regarded as a separate sample. We transform [batch_size, seq_len, height, width, num_channels] to
            # [batch_size*seq_len, height, width, num_channels]
            batch_size, seq_len, height, width, num_channels = self.input_layer.shape
            non_temporal_input_dims = [-1, height, width, num_channels]
            self.input_layer = tf.reshape(self.input_layer, non_temporal_input_dims)
            self.build_network()

            # Shape of [batch_size*seq_len, cnn_height, cnn_width, num_filters]
            batch_seq, cnn_height, cnn_width, num_filters = self.model_output_raw.shape.as_list()
            self.model_output_flat = tf.reshape(self.model_output_raw, [-1, cnn_height*cnn_width*num_filters])

            # Stack a dense layer to set CNN representation size.
            # Densely connected layer with <num_hidden_units> output neurons.
            # Output Tensor Shape: [batch_size, num_hidden_units]
            self.model_output_flat = tf.layers.dense(inputs=self.model_output_flat, units=self.config['num_hidden_units'], activation=tf.nn.relu)
            self.model_output = tf.reshape(self.model_output_flat, [batch_size, -1, self.config['num_hidden_units']])


class RNNModel(Model):
    """
    Recurrent neural network for sequence modeling.
    - Accepts inputs of rank 3 where a mini-batch has shape of [batch_size, seq_len, feature_size].

    """
    def __init__(self, config, placeholders, mode):
        super().__init__(config, placeholders, mode)


    def build_network(self):
        """
        Creates LSTM cell(s) and recurrent model.
        """
        with tf.variable_scope("recurrent", reuse=self.reuse, initializer=self.initializer, regularizer=None):
            rnn_cells = []
            for i in range(self.config['num_layers']):
                rnn_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=self.config['num_hidden_units']))

            if self.config['num_layers'] > 1:
                # Stack multiple cells.
                self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells=rnn_cells, state_is_tuple=True)
            else:
                self.rnn_cell = rnn_cells[0]

            self.model_output_raw, self.rnn_state = tf.nn.dynamic_rnn(cell=self.rnn_cell,
                                                                      inputs=self.input_layer,
                                                                      dtype=tf.float32,
                                                                      sequence_length=self.input_seq_len,
                                                                      time_major=False,
                                                                      swap_memory=True)

    def build_graph(self, input_layer=None):
        with tf.variable_scope("rnn_model", reuse=self.reuse, initializer=self.initializer, regularizer=None):
            if input_layer is None:
                # TODO you can feed any image modality if you wish. You need to flatten images such that a mini-batch has shape [batch_size, seq_len, height*width*num_channels].
                raise Exception("Inputs are missing.")
            else:
                self.input_layer = input_layer

            self.build_network()

            # Shape of [batch_size, seq_len, representation_size]
            batch_size, seq_len, representation_size = self.model_output_raw.shape.as_list()

            self.model_output = self.model_output_raw
            self.model_output_flat = tf.reshape(self.model_output_raw, [-1, representation_size])
