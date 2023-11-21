import numpy as np
import h5py

def load_dataset(ds_name):
    ds_name = ds_name.lower()
    file_dir = "datasets/{}".format(ds_name)
    train_file = "{}/train_{}.h5".format(file_dir, ds_name)
    test_file = "{}/test_{}.h5".format(file_dir, ds_name)
    if(ds_name.count("cats")):
        train_dataset = h5py.File(train_file, "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File(test_file, "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes

        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def import_data(data_set_name, do_log = False):
    training_set_data, training_set_labels, test_set_data, test_set_labels, classes = load_dataset(data_set_name)
    training_set_flatten = training_set_data.reshape(training_set_data.shape[0], -1).T
    test_set_flatten = test_set_data.reshape(test_set_data.shape[0], -1).T
    if(do_log):
        print_data_set_info(data_set_name, training_set_data, training_set_labels, test_set_data, test_set_labels, training_set_flatten, test_set_flatten)
    
    data_dict = {}
    data = [training_set_data, training_set_labels, test_set_data, test_set_labels, training_set_flatten, test_set_flatten]
    keys = ["Training Set Data", "Training Set Labels", "Test Set Data", "Test Set Labels", "Flattened Training Set", "Flattened Test Set"]
    for i in range(len(data)):
        data_dict.update({keys[i]:data[i]})
    return(data_dict)

def print_data_set_info(ds_name, X_train, Y_train, X_test, Y_test, X_train_flat, X_test_flat):
    m_train = X_train.shape[0]
    m_test = X_test.shape[0]
    num_px = X_train.shape[2]
    print("\n************** LOG **************")
    print("Data Set Name: {}".format(ds_name))
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("Training Set Shape: " + str(X_train.shape))
    print ("Training Label Shape: " + str(Y_train.shape))
    print ("Test Set Shape: " + str(X_test.shape))
    print ("Test Labels Shape: " + str(Y_test.shape))                
    print ("Flattened Training Set Shape: " + str(X_train_flat.shape))
    print ("Flattened Test Set Shape: " + str(X_test_flat.shape))
    print("************** LOG **************\n")

