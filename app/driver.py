import matplotlib.pyplot as plt
import sys
import os
from PyQt5.QtCore import Qt, pyqtSignal, QThread
sys.path.insert(1, os.getcwd())
from models import ff_neuralnet, linear_classifier
import gui
from utils import sys_utils

"""
    Implementing interface between UI and state machines
"""
class Driver(QThread):

    signal = pyqtSignal(int, float)

    def __init__(self):
        super().__init__()
        self.model_frames = []
        self.model_frames.append(ModelFrame())

    def load_dataset(self, ds_name, num_batches, frame):
        did_load = False
        msg = ""
        if(sys_utils.check_dataset_existence(ds_name)):
            data = sys_utils.import_data(ds_name, num_batches)
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
                         init_type="scalar", iters=3000):
        loaded = False
        if(model_type in sys_utils.get_available_models()):
            self.model_frames[frame].init_model(model_type, layer_dims, 
                         hidden_fn, output_fn, lrn_rate, init_type, iters)
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
    
    def run(self):
        self.model_frames[0].train(self.signal)
        #return msg


"""
    State Machine. Each model frame is a model and a dataset, and the functions to use them
"""

class ModelFrame():
    def __init__(self):
        self.model = ""
        self.dataset = ""
        self.training_iters = 3000

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
    def init_model(self, model_type, layer_dims, hidden_fn, output_fn, lrn_r8, init_type, iters):
        self.training_iters = iters
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

    def train(self, signal):
        costs = self.model.train(self.dataset.get("Batched Training Set"), self.dataset.get("Batched Training Labels"), self.training_iters, signal)
        return costs

    def get_model_info(self):
        if(self.model == ""):
            return
        else:
            model_info = self.model.get_parameters()
            return(self.pretty_format(model_info))
        
    @staticmethod
    def pretty_format(dict):
        formatted_string = ""
        for key,value in dict.items():
            formatted_string = "{}\n{}: {}".format(formatted_string, key, value)
        formatted_string = formatted_string.strip("\n")
        return formatted_string
        
    def frame_info(self):
        return([self.get_ds_info(), self.get_model_info()])
