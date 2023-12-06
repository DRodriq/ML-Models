import numpy as np
import h5py
import os

def check_dataset_existence(dataset_name):
    exists = dataset_name in get_available_datasets()
    return(exists)

def get_available_datasets():
    proj_dir = os.path.dirname(os.path.realpath(__file__))
    ds_folder = proj_dir + "\\..\\datasets\\"
    sub_folders = [name for name in os.listdir(ds_folder) if os.path.isdir(os.path.join(ds_folder, name))]
    return sub_folders

def check_model_existence(model_name):
    return(model_name in get_available_models())

def get_available_models():
    proj_dir = os.path.dirname(os.path.realpath(__file__))
    model_folder = proj_dir + "\\..\\models\\"
    files = [name for name in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, name))]
    for i in range(len(files)):
        files[i] = files[i].replace(".py", "")
    files.remove("ml_model")
    return files

def load_dataset(ds_name):
    if(check_dataset_existence(ds_name) == False):
        return
    ds_name = ds_name.lower()
    file_dir = "datasets/{}".format(ds_name)
    train_file = "{}/train_{}.h5".format(file_dir, ds_name)
    test_file = "{}/test_{}.h5".format(file_dir, ds_name)
    #if(ds_name.count("cats")):
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

def import_data(data_set_name):
    training_set_data, training_set_labels, test_set_data, test_set_labels, classes = load_dataset(data_set_name)
    training_set_flatten = training_set_data.reshape(training_set_data.shape[0], -1).T
    test_set_flatten = test_set_data.reshape(test_set_data.shape[0], -1).T
    training_set_labels.reshape(-1, training_set_data.shape[0]) #
    test_set_labels.reshape(-1,test_set_data.shape[0]) #
    #ds_info = get_dataset_info(data_set_name, training_set_data, training_set_labels, test_set_data, test_set_labels, training_set_flatten, test_set_flatten)
    
    data_dict = {}
    data = [data_set_name, training_set_data, training_set_labels, test_set_data, test_set_labels, training_set_flatten, test_set_flatten]
    keys = ["Dataset Name", "Training Set Data", "Training Set Labels", "Test Set Data", "Test Set Labels", "Flattened Training Set", "Flattened Test Set"]
    for i in range(len(data)):
        data_dict.update({keys[i]:data[i]})

    return(data_dict)

def format_ds_info(data_dict):
    m_train = data_dict.get("Training Set Data").shape[0]
    m_test = data_dict.get("Test Set Data").shape[0]
    num_px = data_dict.get("Training Set Data").shape[2]
    ds_info = (
                "Dataset Name: {}\n".format(data_dict.get("Dataset Name")) + 
                "Number of examples: " + str(m_train) + "\n" +
                "Number of testing examples: " + str(m_test) + "\n" + 
                "Each item is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)" + "\n" + 
                "Training Set Shape: " + str(data_dict.get("Training Set Data").shape)  + "\n" + 
                "Training Label Shape: " + str(data_dict.get("Training Set Labels").shape) + "\n" + 
                "Test Set Shape: " + str(data_dict.get("Test Set Data").shape) + "\n" + 
                "Test Labels Shape: " + str(data_dict.get("Test Set Labels").shape) + "\n" + 
                "Flattened Training Set Shape: " + str(data_dict.get("Flattened Training Set").shape) + "\n" + 
                "Flattened Test Set Shape: " + str(data_dict.get("Flattened Test Set").shape) + "\n"
        )   

    return ds_info
