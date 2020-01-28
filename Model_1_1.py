from keras import Sequential
from keras.backend import mean, std
from keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from matplotlib import pyplot
from LoadData_1_1 import load_dataset, plot_variables_distribution, scale_data


def fit_model(train_x, train_y, filter_size, kernel_size, verbose=0, epochs=10, batch_size=32):
    n_time_steps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # The Sequential model is a linear stack of layers.
    cnn_model = Sequential()
    # filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    # kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
    # activation: Activation function to use (see activations). If you don't specify anything, no activation is applied
    #                                        (ie. "linear" activation: a(x) = x).
    '''
    Input shape
        3D tensor with shape: (batch, steps, channels)
    Output shape
        3D tensor with shape: (batch, new_steps, filters) steps value might have changed due to padding or strides.
    '''
    cnn_model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu',
                         input_shape=(n_time_steps, n_features)))
    cnn_model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu'))
    # rate: float between 0 and 1. Fraction of the input units to drop.
    cnn_model.add(Dropout(0.5))
    # pool_size: Integer, size of the max pooling windows.
    cnn_model.add(MaxPooling1D(pool_size=2))
    # Flattens the input. Does not affect the batch size.
    cnn_model.add(Flatten())
    # Just your regular densely-connected NN layer.
    cnn_model.add(Dense(100, activation='relu'))
    # units: Positive integer, dimensionality of the output space.
    # activation: Activation function to use (see activations). If you don't specify anything, no activation is applied
    #                                        (ie. "linear" activation: a(x) = x).
    cnn_model.add(Dense(units=n_outputs, activation='softmax'))
    '''
    A loss function (or objective function, or optimization score function) is one of the two parameters required to 
    compile a model:
    loss: String (name of objective function) or objective function or Loss instance. See losses.         
          If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or 
          a list of losses. 
          The loss value that will be minimized by the model will then be the sum of all individual losses.
          
    optimizer: String (name of optimizer) or optimizer instance. See optimizers.
    
    metrics: List of metrics to be evaluated by the model during training and testing. 
             Typically you will use metrics=['accuracy']. 
             To specify different metrics for different outputs of a multi-output model,
             you could also pass a dictionary, such as metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}.
             You can also pass a list (len = len(outputs)) of lists of metrics such as metrics=[['accuracy'], 
             ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']].
    '''
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    '''
    The model is fit for a fixed number of epochs, in this case 10, 
    and a batch size of 32 samples will be used, where 32 windows of data will be exposed to the model 
    before the weights of the model are updated.
    '''
    # 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
    cnn_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return cnn_model


def evaluate_model(model, test_x, test_y, batch_size):
    # evaluate model
    _, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    return accuracy


# summarize scores
def summarize_results(scores, standardize, filter_size, kernel_size):
    print(scores, standardize)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        _mean, _std = mean(scores[i]), std(scores[i])
        print('standardize=%d, filter_size=%d, kernel_size=%d : %.3f%% (+/-%.3f)' %
              (standardize, filter_size, kernel_size, _mean, _std))
    # box plot of scores
    pyplot.boxplot(scores, labels='standardize=%d, filter_size=%d, kernel_size=%d)' %
                                  (standardize, filter_size, kernel_size))
    pyplot.savefig('exp_cnn_summary.png')


# run an experiment
def run_experiment(standardize_list, n_filters, n_kernels, repeats=10):
    # load data
    train_x, train_y, test_x, test_y = load_dataset()
    plot_variables_distribution(train_x)
    verbose, epochs, batch_size = 0, 10, 32
    # test each standardize
    all_scores = list()
    for standardize in standardize_list:
        # test for each filter
        for filter_size in n_filters:
            # test for each kernel
            for kernel_size in n_kernels:
                # repeat experiment
                scores = list()
                for r in range(repeats):
                    train_x, test_x = scale_data(train_x, test_x, standardize)
                    model = fit_model(train_x, train_y, filter_size, kernel_size, verbose, epochs, batch_size)
                    score = evaluate_model(model, test_x, test_y, batch_size)
                    score = score * 100
                    print('>standardize=%s filter_size=%s kernel_size=%s #%d: %.3f' %
                          (standardize, filter_size, kernel_size, r+1, score))
                    scores.append(score)
                all_scores.append(scores)
                # summarize results
                summarize_results(all_scores, standardize, filter_size, kernel_size)
