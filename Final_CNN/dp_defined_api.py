from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf


class Network():

    def __init__(self, train_batch_size, test_batch_size, pooling_scale,
                 dropout_rate, base_learning_rate, decay_rate,
                 optimizeMethod='adam', save_path='model/default.ckpt'):

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Train Related
        self.optimizeMethod = optimizeMethod
        self.dropout_rate = dropout_rate
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate

        # Hyper Parameters
        self.conv_config = []  # list of dict
        self.fc_config = []  # list of dict
        self.conv_weights = []
        self.conv_biases = []
        self.fc_weights = []
        self.fc_biases = []
        self.pooling_scale = pooling_scale
        self.pooling_stride = pooling_scale

        # Graph Related
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None

        # Summary Related
        self.writer = None
        self.merged = None
        self.train_summaries = []
        self.test_summaries = []

        # Save model
        self.saver = None
        self.save_path = save_path

    def add_conv(self, *, patch_size, in_depth, out_depth, activation='relu', pooling=False, name):
        '''
        This function does not define operations in the graph, but only store config in self.conv_layer_config
        '''
        self.conv_config.append(
            {
                'patch_size': patch_size,
                'in_depth': in_depth,
                'out_depth': out_depth,
                'activation': activation,
                'pooling': pooling,
                'name': name
            })

        with tf.name_scope(name):
            weights = tf.Variable(
                tf.truncated_normal([patch_size, patch_size, in_depth, out_depth], stddev=0.1), name=name + '_weights')

            biases = tf.Variable(
                tf.constant(0.1, shape=[out_depth]), name=name + '_biases')

            self.conv_weights.append(weights)
            self.conv_biases.append(biases)

    def add_fc(self, *, in_num_nodes, out_num_nodes, activation='relu', name):
        '''
        add fc layer config to self.fc_layer_config
        '''
        self.fc_config.append(
            {
                'in_num_nodes': in_num_nodes,
                'out_num_nodes': out_num_nodes,
                'activation': activation,
                'name': name
            })

        with tf.name_scope(name):
            weights = tf.Variable(
                tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=0.1))

            biases = tf.Variable(
                tf.constant(0.1, shape=[out_num_nodes]))  # initialize the bias as 0.1

            self.fc_weights.append(weights)
            self.fc_biases.append(biases)
            self.train_summaries.append(tf.summary.histogram(
                str(len(self.fc_weights)) + '_weights', weights))
            self.train_summaries.append(tf.summary.histogram(
                str(len(self.fc_biases)) + '_biases', biases))

    def apply_regularization(self, _lambda):
        # L2 regularization for the fully connected parameters
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
            '''
            l2_loss(t, name = None), output = sum(t ** 2) / 2
            '''
        return _lambda * regularization

    def define_inputs(self, *, train_samples_shape, train_labels_shape, test_samples_shape):

        with tf.name_scope('inputs'):
            self.tf_train_samples = tf.placeholder(
                tf.float32, shape=train_samples_shape, name='tf_train_samples')
            self.tf_train_labels = tf.placeholder(
                tf.float32, shape=train_labels_shape, name='tf_train_labels')
            self.tf_test_samples = tf.placeholder(
                tf.float32, shape=test_samples_shape, name='tf_test_samples')

    def define_model(self):
        '''
        Define my calculating graph
        '''
        def model(data_flow, train=True):
            '''
            @data: original inputs
            @return: logits
            '''
            # define convolution layers
            for i, (weights, biases, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_config)):
                # 'name' is in the 'conv_config' dict
                with tf.name_scope(config['name'] + '_model'):

                    with tf.name_scope('convolution'):
                        # default 1,1,1,1 stride and SAME padding, the input
                        # data-format is "NHWC", stride is the sliding window
                        # for each dimension of input
                        data_flow = tf.nn.conv2d(
                            data_flow, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
                        data_flow = data_flow + biases
                        if not train:
                            self.visualize_filter_map(
                                data_flow, how_many=config['out_depth'], display_size=32 // (i // 2 + 1), name=config['name'] + '_conv')
                    if config['activation'] == 'relu':
                        data_flow = tf.nn.relu(data_flow)
                        if not train:
                            self.visualize_filter_map(
                                data_flow, how_many=config['out_depth'], display_size=32 // (i // 2 + 1), name=config['name'] + '_relu')
                    else:
                        raise Exception(
                            'Activation Func can only be Relu right now. You passed', config['activation'])

                    if config['pooling']:
                        data_flow = tf.nn.max_pool(
                            data_flow,
                            ksize=[1, self.pooling_scale,
                                   self.pooling_scale, 1],
                            strides=[1, self.pooling_stride,
                                     self.pooling_stride, 1],
                            padding='SAME')
                        if not train:
                            self.visualize_filter_map(
                                data_flow, how_many=config['out_depth'], display_size=32 // (i // 2 + 1), name=config['name'] + '_pooling')

            # Define Fully Connected Layers
            for i, (weights, biases, config) in enumerate(zip(self.fc_weights, self.fc_biases, self.fc_config)):

                if i == 0:  # connection between convolution layer and the fully connected layer
                    shape = data_flow.get_shape().as_list()
                    data_flow = tf.reshape(
                        data_flow, [shape[0], shape[1] * shape[2] * shape[3]])

                with tf.name_scope(config['name'] + '_model'):

                    # Dropout at the last fully connected layer, randomly
                    # dropout the nn output
                    # only dropout at the training step
                    if train and i == len(self.fc_weights) - 1:
                        data_flow = tf.nn.dropout(
                            data_flow, self.dropout_rate, seed=4926)

                    ###
                    data_flow = tf.matmul(data_flow, weights) + biases
                    if config['activation'] == 'relu':
                        dataflow = tf.nn.relu(data_flow)
                    elif config['activation'] is None:
                        pass
                    else:
                        raise Exception(
                            'Activation Func can only be Relu or None right now. You passed', config['activation'])

            return data_flow  # this is the finnal output for the fully connected layer

        # training computation
        logits = model(self.tf_train_samples)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels, logits=logits))

            self.loss += self.apply_regularization(_lambda=5e-4)

            self.train_summaries.append(tf.summary.scalar('Loss', self.loss))

        # Learning rate decay
        global_step = tf.Variable(0)
        '''
        when training a model, it is often recommended to lower the learning rate as the training progresses.
        this function applies an exponential decay function to a provided initial learning rate.
        returns decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        '''
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.base_learning_rate,
            global_step=global_step * self.train_batch_size,
            decay_steps=100,
            decay_rate=self.decay_rate,
            staircase=True
        )

        # Optimizer
        with tf.name_scope('optimizer'):
            if (self.optimizeMethod == 'gradient'):
                self.optimizer = tf.train \
                    .GradientDescentOptimizer(learning_rate) \
                    .minimize(self.loss)
            elif(self.optimizeMethod == 'momentum'):
                self.optimizer = tf.train \
                    .MomentumOptimizer(learning_rate, momentum=0.5) \
                    .minimize(self.loss)
            elif(self.optimizeMethod == 'adam'):
                self.optimizer = tf.train \
                    .AdamOptimizer(learning_rate) \
                    .minimize(self.loss)

        # Predictions for the training, validation and test data
        with tf.name_scope('train'):
            self.train_prediction = tf.nn.softmax(
                logits, name='train_prediction')
        with tf.name_scope('test'):
            self.test_prediction = tf.nn.softmax(
                model(self.tf_test_samples), name='test_predictions')

        self.merged_train_summary = tf.summary.merge(self.train_summaries)
        # self.merged_test_summary = tf.summary.merge(self.test_summaries)

    def run(self, data_iterator, train_samples, train_labels, test_samples, test_labels):
        '''
        define the Session here
        @data_iterator is a function that yields chunk of data
        '''
        # private function
        def print_confusion_matrix(confusionMatrix):
            print('Confusion Matrix')
            for i, line in enumerate(confusionMatrix):
                print(line, line[i] / np.sum(line))

            print('\n')
            a = 0
            for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
                a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
                print(column, column[i] / np.sum(column))
            print('\n', np.sum(confusionMatrix), a)

        self.writer = tf.summary.FileWriter('./board', tf.get_default_graph())

        with tf.Session(graph=tf.get_default_graph()) as session:
            session.run(tf.global_variables_initializer())

            print('Start Training')
            for i, samples, labels in data_iterator(train_samples, train_labels, chunkSize=self.train_batch_size):
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction,
                        self.merged_train_summary],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels})

                self.writer.add_summary(summary, i)
                accuracy, _ = self.accuracy(predictions, labels)
                if i % 50 == 0:
                    print('Minibatch loss at step %d : %f' % (i, l))
                    print('Minibatch accuracy %.1f%%' % accuracy)

            # save the model
            import os
            if os.path.isidr(self.save_path.split('/')[0]):
                save_path = self.saver.save(session, self.save_path)
                print('Model saved in file: %s' % save_path)
            else:
                os.makedirs(self.save_path.split('/')[0])
                save_path = self.saver.save(session, self, save_path)
                print('Model saved in file: %s' % save_path)

            print('Start Testing')
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                print('samples shape', samples.shape)
                # result, summary = session.run(
                #     [self.test_prediction, self.merged_test_summary],
                # feed_dict={self.tf_test_samples: samples,
                # self.tf_test_labels: labels})

                result = self.test_prediction.eval(
                    feed_dict={self.tf_test_samples: samples})

                # self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(
                    result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)

            print('Average Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            print_confusion_matrix(np.add.reduce(confusionMatrices))

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        '''
        @return: accuracy and confusionMatrix as a tuple
        '''
        # mind that right now all the predictions and labels are in
        # [0,0,0,1,0,0,0,0,0] format
        # return the index of the number 1
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(
            _labels, _predictions) if need_confusion_matrix else None
        accuracy = (100.0 * np.sum(_predictions ==
                                   _labels) / predictions.shape[0])
        return accuracy, cm

    def visualize_filter_map(self, tensor, *, how_many, display_size, name):
        print(tensor.get_shape)
        filter_map = tensor[-1]
        print(filter_map.get_shape())
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        print(filter_map.get_shape())
        filter_map = tf.reshape(filter_map, (how_many, display_size, 1))
        print(how_many)
        self.test_summaries.append(tf.image_summary(
            name, tensor=filter_map, max_images=how_many))
