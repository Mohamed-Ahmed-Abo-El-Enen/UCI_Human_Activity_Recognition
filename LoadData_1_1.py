import numpy as np
import pandas as pnd
from matplotlib import pyplot
import keras.utils as utl
from sklearn.preprocessing import StandardScaler


# load a single file as a numpy array
def load_file(filepath):
    data_frame = pnd.read_csv(filepath, header=None, delim_whitespace=True)
    return data_frame.values


# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    x = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix+group+'/y_'+group+'.txt')
    return x, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    train_x, train_y = load_dataset_group('train', prefix+'UCI HAR Dataset/')
    print(train_x.shape, train_y.shape)
    # load all test
    test_x, test_y = load_dataset_group('test', prefix+'UCI HAR Dataset/')
    print(test_x.shape, test_y.shape)
    # zero-offset class values
    train_y = train_y - 1
    test_y = test_y - 1
    # one hot encode y
    train_y = utl.to_categorical(train_y)
    test_y = utl.to_categorical(test_y)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y


# plot a histogram of each variables in the dataset
def plot_variables_distribution(train_x):
    # remove overlap
    cut = int(train_x.shape[1] / 2)
    long_x = train_x[:, -cut:, :]
    # flatten windows
    long_x = long_x.reshape((long_x.shape[0] * long_x.shape[1], long_x.shape[2]))
    print(long_x.shape)
    pyplot.figure()
    x_axis = None
    for i in range(long_x.shape[1]):
        ax = pyplot.subplot(long_x.shape[1], 1, i+1, sharex=x_axis)
        ax.set_xlim(-1, 1)
        if i == 0:
            x_axis = ax
        pyplot.hist(long_x[:, i], bins=100)
    pyplot.show()


# standardize data
def scale_data(train_x, test_x, standardize):
    # remove overlap
    cut = int(train_x.shape[1] / 2)
    long_x = train_x[:, -cut:, :]
    # flatten windows
    long_x = long_x.reshape((long_x.shape[0] * long_x.shape[1], long_x.shape[2]))
    # flatten train and test
    flatten_train_x = train_x.reshape((train_x.shape[0] * train_x.shape[1], train_x.shape[2]))
    flatten_test_x = test_x.reshape((test_x.shape[0] * test_x.shape[1], test_x.shape[2]))
    # standardize
    if standardize:
        s = StandardScaler()
        # fit on training data
        s.fit(long_x)
        # apply to training and testing data
        long_x = s.transform(long_x)
        flatten_train_x = s.transform(flatten_train_x)
        flatten_test_x = s.transform(flatten_test_x)
    # reshape
    flatten_train_x = flatten_train_x.reshape(train_x.shape)
    flatten_test_x = flatten_test_x.reshape(test_x.shape)
    return flatten_train_x, flatten_test_x

