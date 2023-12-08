import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(1, os.getcwd())
from utils import sys_utils
import time
from models import ff_neuralnet
from abc import ABC, abstractmethod

class ModelStageInterface(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def set_dataset(self, c_log) -> tuple[bool, str]:
        pass

    @abstractmethod
    def clear_dataset(self) -> None:
        pass

    @abstractmethod
    def dataset_isloaded(self) -> bool:
        pass

    @abstractmethod
    def set_model(self, string, list, dictionary) -> tuple[bool, str]:
        pass

    @abstractmethod
    def clear_model(self) -> None:
        pass

    @abstractmethod
    def model_isloaded(self) -> bool:
        pass

    @abstractmethod
    def model_hastrained(self) -> bool:
        pass

    @abstractmethod
    def do_train(self, int, signal) -> tuple[bool, str]:
        pass

    @abstractmethod
    def do_test(self) -> tuple[bool, str]:
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        pass

    @abstractmethod
    def get_stage_info(self) -> dict:
        pass

    @abstractmethod
    def save_results(self) -> None:
        pass


class ModelStage(ModelStageInterface):
    def __init__(self, console_log=False):
        self.model = None
        self.dataset = None
        self.training_costs = []
        self.last_training_accuracy = 0
        self.last_test_accuracy = 0

        self.ds_trained_on = []
        self.training_iterations_run = 0
        self.time_spent_training = 0
        self.console_log = console_log
    
    def set_dataset(self, ds_name, batches=1):
        self.dataset = sys_utils.import_data(ds_name, batches)
        msg = ""
        if(self.model_isloaded()):
            if(not(self.model.layer_dims[0] == self.dataset.get("Flattened Training Set").shape[0])):
                msg = "***WARNING!*** Current model input layer does not match training set feature vector size!"
        msg = msg + "Dataset {} successfully loaded".format(ds_name)
        return (True, msg)
    
    def clear_dataset(self):
        self.dataset = None

    def dataset_isloaded(self):
        isLoaded = False
        if(self.dataset is not None):
            isLoaded = True
        return isLoaded
    
    def get_dataset_info(self):
        msg = ""
        if(self.dataset_isloaded()):
            msg = sys_utils.format_ds_info(self.dataset)
        return msg
  
    def set_model(self, model_type, layer_dims, **kwargs):
        successful_load = False
        msg = ""
        if(self.dataset_isloaded()):
            if(not(layer_dims[0] == self.dataset.get("Flattened Training Set").shape[0])):
                msg = "**WARNING**! Model input layer does not match data training set feature vector size: {}".format(
                    self.dataset.get("Flattened Training Set").shape[0])
                return(successful_load, msg)
        if(model_type == "ff_neuralnet"):
            self.model = ff_neuralnet.FF_NeuralNetwork(layers_dims=layer_dims, **kwargs)
            successful_load = True
            msg = "{} model initialized".format(model_type)
        elif(model_type == "linear_classifier"):
            pass
            #self.model = linear_classifier()
        return(successful_load, msg)
    
    def clear_model(self):
        self.model = None
        self.training_iterations_run = 0
        self.time_spent_training = 0
        self.training_costs = []
        self.last_training_accuracy = 0
        self.last_test_accuracy = 0
        self.ds_trained_on = []

    def model_isloaded(self):
        isLoaded = False
        if(self.model is not None):
            isLoaded = True
        return isLoaded

    def model_hastrained(self):
        if(self.training_iterations_run != 0):
            return(True)
        return False

    def do_train(self, training_iterations=3000, signal=None):
        did_train = False
        if(self.model_isloaded()):
            if(self.dataset_isloaded()):
                start = time.time()
                self.training_costs = self.training_costs + self.model.train(
                    self.dataset.get("Batched Training Set"), 
                    self.dataset.get("Batched Training Labels"),
                    training_iterations, signal, self.console_log)   
                execution_time = round(time.time() - start, 2)
                self.training_iterations_run = self.training_iterations_run + training_iterations
                self.time_spent_training = self.time_spent_training + execution_time 
                did_train = True
                self.ds_trained_on = self.ds_trained_on + [self.dataset.get("Dataset Name")]
                msg = "Model trained in {}s".format(execution_time)
            else:
                msg = "Cannot train. No dataset loaded"
        else:
            msg = "Cannot train. No model loaded"
        return did_train, msg 

    def do_test(self):
        did_Test = False
        if(self.model_isloaded()):
            if(self.dataset_isloaded()):
                self.last_training_accuracy = self.model.test_model(self.dataset.get("Flattened Training Set"), self.dataset.get("Training Set Labels"), self.console_log)
                self.last_test_accuracy = self.model.test_model(self.dataset.get("Flattened Test Set"), self.dataset.get("Test Set Labels"), self.console_log)
                msg = "Training Accuracy: {} | Test Accuracy {}".format(self.last_training_accuracy, self.last_test_accuracy)
                did_Test = True
                print("msg")
            else:
                msg = "Cannot test model. No model loaded"
        else:
            msg = "Cannot test model. No dataset loaded"           
        return did_Test, msg
    
    def get_model_info(self):
        #is_info=False
        m_info = ""
        if(self.model_isloaded()):
            #is_info = True
            m_info = self.model.get_parameters()
        return m_info
    
    def get_stage_info(self):
        is_info = False
        s_info = {}
        is_info, m_info = self.get_model_info()
        if(is_info):
            s_info = m_info
            s_info.update({"Model Trained": False})
            if(self.model_hastrained()):
                s_info.update({"Model Trained": True})
                s_info.update({"Datasets Trained On": self.ds_trained_on})
                s_info.update({"Training Set Accuracy": self.last_training_accuracy})
                s_info.update({"Test Set Accuracy": self.last_test_accuracy})
                s_info.update({"Costs": self.training_costs})
                s_info.update({"Training Iterations": self.training_iterations_run})
                s_info.update({"Time Spent Training": self.time_spent_training})
                s_info.update({"Average Training Iterations / Second": round(self.training_iterations_run / self.time_spent_training)})
            is_info = True
        return is_info, m_info
    
    def save_model(self, name):
        sys_utils.save_datastructure(self.model, name)

    def load_model(self, name):
        self.clear_model()
        self.model = sys_utils.load_datastructure(name)
    
    def save_results(self):
        is_info, s_info = self.get_stage_info()
        if(is_info):
            sys_utils.post_process(s_info)