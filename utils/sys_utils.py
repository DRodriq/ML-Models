import numpy as np
import h5py
import os
import datetime
import matplotlib.pyplot as plt
import datetime
import pickle

def check_dataset_existence(dataset_name):
    exists = dataset_name in get_available_datasets()
    return(exists)

def get_available_datasets():
    proj_dir = os.path.dirname(os.path.realpath(__file__))
    ds_folder = proj_dir + "\\..\\datasets\\"
    sub_folders = [name for name in os.listdir(ds_folder) if os.path.isdir(os.path.join(ds_folder, name))]
    return sub_folders

def check_model_type_existence(model_name):
    return(model_name in get_available_model_types())

def get_available_model_types():
    proj_dir = os.path.dirname(os.path.realpath(__file__))
    model_folder = proj_dir + "\\..\\models\\"
    files = [name for name in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, name))]
    for i in range(len(files)):
        files[i] = files[i].replace(".py", "")
    files.remove("ml_model")
    files.remove("model_stage")
    return files

def get_saved_models():
    proj_dir = os.path.dirname(os.path.realpath(__file__))
    model_folder = proj_dir + "\\..\\results\\saved_models\\"
    files = [name for name in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, name))]
    return files

def check_saved_model_existence(model_name):
    return(model_name in get_saved_models())

def load_dataset(ds_name):
    if(check_dataset_existence(ds_name) == False):
        return
    ds_name = ds_name.lower()
    file_dir = "datasets/{}".format(ds_name)
    train_file = "{}/train_{}.h5".format(file_dir, ds_name)
    test_file = "{}/test_{}.h5".format(file_dir, ds_name)

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

def import_data(data_set_name, batches=1):
    training_set_data, training_set_labels, test_set_data, test_set_labels, classes = load_dataset(data_set_name)

    training_set_flatten = training_set_data.reshape(training_set_data.shape[0], -1).T
    test_set_flatten = test_set_data.reshape(test_set_data.shape[0], -1).T
    training_set_labels.reshape(-1, training_set_data.shape[0]) 
    test_set_labels.reshape(-1,test_set_data.shape[0]) 

    training_set_flatten = training_set_flatten / np.amax(training_set_flatten)
    test_set_flatten = test_set_flatten / np.amax(test_set_flatten)
    
    X_train_batched = batch(training_set_flatten, batches)
    y_train_batched = batch(training_set_labels, batches)

    data_dict = {}
    data = [data_set_name, training_set_data, training_set_labels, test_set_data, test_set_labels, training_set_flatten, test_set_flatten, X_train_batched, y_train_batched]
    keys = ["Dataset Name", "Training Set Data", "Training Set Labels", "Test Set Data", 
            "Test Set Labels", "Flattened Training Set", "Flattened Test Set", "Batched Training Set", "Batched Training Labels"]
    for i in range(len(data)):
        data_dict.update({keys[i]:data[i]})

    return(data_dict)

def format_ds_info(data_dict):
    m_train = data_dict.get("Training Set Data").shape[0]
    m_test = data_dict.get("Test Set Data").shape[0]
    num_px = data_dict.get("Training Set Data").shape[2]
    ds_info = (
                "Dataset Name: {}\n".format(data_dict.get("Dataset Name")) + 
                "Training Set Size: " + str(m_train) + "\n" +
                "Test Set Size: " + str(m_test) + "\n" + 
                "Each item is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)" + "\n" + 
                "Training Set Shape: " + str(data_dict.get("Training Set Data").shape)  + "\n" + 
                "Training Label Shape: " + str(data_dict.get("Training Set Labels").shape) + "\n" + 
                "Test Set Shape: " + str(data_dict.get("Test Set Data").shape) + "\n" + 
                "Test Labels Shape: " + str(data_dict.get("Test Set Labels").shape) + "\n" + 
                "Flattened Training Set Shape: " + str(data_dict.get("Flattened Training Set").shape) + "\n" + 
                "Flattened Test Set Shape: " + str(data_dict.get("Flattened Test Set").shape) + "\n"
                "Number of Batches: " + str(len(data_dict.get("Batched Training Set"))) + "\n" + 
                "Batch Size: " + str(data_dict.get("Batched Training Set")[0].shape[1]) + "\n"
                "Batched Training Set Size: " + str(data_dict.get("Batched Training Set")[0].shape)
        )   

    return ds_info

def batch(X, n):
    """
    Split the dataset X into n batches.
    Parameters:
    - X: NumPy array, shape (features, samples)
    - n_batches: Number of batches

    Returns:
    - List of batches, where each batch is a NumPy array
    """
    samples_per_batch = X.shape[1] // n
    # Reshape the array to have the samples in the first dimension
    X = X.T
    batches = [X[i * samples_per_batch : (i + 1) * samples_per_batch, :].T for i in range(n - 1)]
    # The last batch might have a different size if the total number of samples is not divisible by n_batches
    batches.append(X[(n - 1) * samples_per_batch :, :].T)
    #for i in range(len(batches)):
    #    np.reshape(batches, (batches[i].shape[0], 1))
    return batches

def post_process(params):
    now = datetime.datetime.now()
    timestamp = now.strftime("%m/%d/%Y-%H:%M:%S")
    file_friendly_ts = now.strftime("%m-%d-%Y-%H_%M-%S")
    proj_dir = os.path.dirname(os.path.realpath(__file__))
    results_folder = proj_dir + "\\..\\results\\"
    results_file = results_folder + "logs\\ff_nn.log"
    costs_file_name = "plots\\nn_costs-" + file_friendly_ts + ".png"
    costs_file = results_folder + costs_file_name

    entry_title = "********** {} Stage Stats **********\n".format(timestamp)

    if("Costs", True in params.items()):
        plt.plot(params.get("Costs"))
        plt.title(costs_file_name.replace(".png", ''))
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig(costs_file)
        f = open(results_file, "a")

    results = ""
    for key, value in params.items():
        if(key != "Costs"):
            results = results + "{}: {}\n".format(key, str(value))

    other_results_info = "Costs Plot File: {}\n".format(costs_file_name)
    f.write(entry_title)
    f.write(results)
    f.write(other_results_info)
    f.write("********************************************\n")
   
def save_datastructure(data_structure, file_name):
    proj_dir = os.path.dirname(os.path.realpath(__file__))
    results_folder = proj_dir + "\\..\\results\\"
    full_path = results_folder + "saved_models\\" + file_name
    file = open(full_path, 'w+b')
    pickle.dump(data_structure, file)

def load_datastructure(name):
    if(check_saved_model_existence(name)):
        proj_dir = os.path.dirname(os.path.realpath(__file__))
        results_folder = proj_dir + "\\..\\results\\"
        ds_file = results_folder + "saved_models\\" + name
        with open(ds_file, 'rb') as file:
            loaded_data = pickle.load(file)
        return(loaded_data)