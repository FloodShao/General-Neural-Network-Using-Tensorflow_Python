if __name__ == '__main__':
    import preProcessing
    from dp_defined_api import Network

    train_samples, train_labels = preProcessing._train_samples, preProcessing._train_labels
    test_samples, test_labels = preProcessing._test_samples, preProcessing._test_labels

    print('Train_set', train_samples.shape, train_labels.shape)
    print('Test_set', test_samples.shape, test_labels.shape)

    image_size = preProcessing.image_size
    num_labels = preProcessing.num_labels
    num_channels = preProcessing.num_channels

    train_batch = 64
    test_batch = 500
    pooling_scaleg = 2

    '''
    Here define the data_iterator and pass to the run function later
    '''
    def get_chunk(samples, labels, chunkSize):
        '''
        Iterator/Generator: get a batch of data
        '''
        if len(samples) != len(labels):
            raise Exception('Length of samples and labels must equal')

        stepStart = 0  # initial step
        i = 0
        while stepStart < len(samples):
            stepEnd = stepStart + chunkSize
            if stepEnd < len(samples):
                yield i, samples[stepStart: stepEnd], labels[stepStart: stepEnd]
                i += 1
            stepStart = stepEnd

    net = Network(
        train_batch_size=64, test_batch_size=500, pooling_scale=2,
        dropout_rate=0.9, base_learning_rate=0.001, decay_rate=0.99,
    )

    net.define_inputs(
        train_samples_shape=(train_batch, image_size,
                             image_size, num_channels),
        train_labels_shape=(train_batch, num_labels),
        test_samples_shape=(test_batch, image_size, image_size, num_channels))

    # convolution layer
    net.add_conv(patch_size=3, in_depth=num_channels, out_depth=32,
                 activation='relu', pooling=False, name='conv1')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32,
                 activation='relu', pooling=True, name='conv2')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32,
                 activation='relu', pooling=False, name='conv3')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32,
                 activation='relu', pooling=True, name='conv4')

    # fullyconnected layer
    # 4 = pooling twice, each pooling reduce half
    # 16 = conv4 out_depth
    # because in each convolution layer, we use the padding = 'SAME', so after
    # convoluted with filter, the output size is the same as input
    net.add_fc(
        in_num_nodes=(image_size // 4) * (image_size // 4) * 32, out_num_nodes=128, activation='relu', name='fc1')
    net.add_fc(
        in_num_nodes=128, out_num_nodes=10, activation=None, name='fc2')

    net.define_model()

    net.run(get_chunk, train_samples, train_labels, test_samples, test_labels)

else:
    raise Exception(
        'main.py: Should Not Be Imported !!! Must Run by "python main.py"')
