import threading
import time
import matplotlib.pyplot as plt
from queue import Queue
import sys
import os
sys.path.insert(1, os.getcwd())
from models import ff_neuralnet, linear_classifier
import gui
from utils import sys_utils

"""
    Implementing interface between UI and state machines
"""
class Driver():
    def __init__(self):
        self.model_frames = []
        self.model_frames.append(ModelFrame())
        self.q = Queue()

    def load_dataset(self, ds_name, frame):
        did_load = False
        msg = ""
        if(sys_utils.check_dataset_existence(ds_name)):
            data = sys_utils.import_data(ds_name)
            if(self.model_frames[frame].ds_isloaded() == ds_name):
                msg = "Dataset '{}' already loaded in frame {}".format(ds_name, str(frame))
            else:
                did_load = True
                self.model_frames[frame].add_ds(data)
                msg = self.model_frames[frame].get_ds_info()
        else:
            msg = "Dataset '{}' does not exist".format(ds_name)
        return did_load, msg
    
    def validate_ds(self, frame):
        return(self.model_frames[frame].ds_isloaded())
    
    def validate_model(self, frame):
        return(self.model_frames[frame].model_isloaded())
    
#def __init__(self, layers_dims, hidden_activation_func="sigmoid", output_activation_func="sigmoid", learning_rate=0.01, init_type="scalar", log ="none"):
    def initialize_model(self, 
                         model_type, layer_dims, frame, lrn_rate=.03, 
                         hidden_fn="tanh", output_fn="sigmoid", 
                         init_type="scalar"):
        loaded = False
        if(model_type in sys_utils.get_available_models()):
            self.model_frames[frame].init_model(model_type, layer_dims,lrn_rate, 
                         hidden_fn, output_fn, init_type)
            loaded = True
            msg = self.model_frames[frame].get_model_info()
        else:
            msg = "Model Type '{}' does not exist".format(model_type)
        return loaded, msg
    
    def clear_model(self, frame):
        self.model_frames[frame].clear_model()

    def add_frame(self):
        self.model_frames.append(ModelFrame())

    def delete_frame(self, frame):
        self.model_frames.pop(frame)

    def clear_dataset(self, frame):
        self.model_frames[frame].clear_ds()

    def get_frame_info(self, frame):
        return(self.model_frames[frame].frame_info())
    
    def train_model(self, frame):
        msg = self.model_frames[frame].train()
        return msg


"""
    State Machine. Each model frame is a model and a dataset, and the functions to use them
"""

class ModelFrame():
    def __init__(self):
        self.model = ""
        self.dataset = ""

    def add_ds(self, ds):
        self.dataset = ds

    def clear_ds(self):
        self.dataset = ""

    def ds_isloaded(self):
        if(self.dataset == ""):
            return(False)
        return(self.dataset.get("Dataset Name"))

    def get_ds_info(self):
        if(self.dataset == ""):
            return ""
        else:
            return(sys_utils.format_ds_info(self.dataset))
        
#def __init__(self, layers_dims, hidden_activation_func="sigmoid", output_activation_func="sigmoid", learning_rate=0.01, init_type="scalar", log ="none"):       
    def init_model(self, model_type, layer_dims, hidden_fn, output_fn, lrn_r8, init_type):
        if(model_type == "ff_neuralnet"):
            self.model = ff_neuralnet.FF_NeuralNetwork(layer_dims, hidden_fn, output_fn, lrn_r8, init_type)
        elif(model_type == "linear_classifier"):
            self.model = linear_classifier.LinearClassifier()

    def model_isloaded(self):
        isloaded = True
        if(self.model == ""):
            isloaded = False
        return isloaded

    def clear_model(self):
        self.model = ""

    def train(self):
        costs = self.model.train(self.dataset.get("Flattened Training Set"), self.dataset.get("Training Set Labels"), 3000)
        return costs

    def get_model_info(self):
        if(self.model == ""):
            return
        else:
            model_info = self.model.get_hyperparameters()
            return(self.pretty_format(model_info))
        
    @staticmethod
    def pretty_format(dict):
        formatted_string = ""
        for key,value in dict.items():
            formatted_string = "{}\n{}: {}".format(formatted_string, key, value)
        formatted_string = formatted_string.strip("\n")
        return formatted_string
        
    def frame_info(self):
        return([self.get_ds_info(),])


if __name__ == '__main__':
    app = gui.QApplication(sys.argv)
    app_window = gui.ModelTesterApp()
    app_window.log("Session Started: " + app_window.session_name, "GUI")
    q = Queue()
    app_window.log("Application Ready: " + app_window.session_name, "DRIVER")

    cmd = command.LoadDatasetCommand()
    
    """
    model_thread = threading.Thread(target=ff_neuralnet.thread_run, args=(q,), daemon=None)
    #TypeError: run() missing 7 required positional arguments: 'data_set', 'do_standardize_data', 'nn_dims', 'act_fn', 'init_type', 'lrn_rate', and 'training_iterations'
    model_thread.start()
    time.sleep(1)
    while(model_thread.is_alive() == True):
        print(q.get())
    """
    sys.exit(app.exec_())
