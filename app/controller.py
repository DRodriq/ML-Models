import matplotlib.pyplot as plt
import sys
import os
from PyQt5.QtCore import Qt, pyqtSignal, QThread
sys.path.insert(1, os.getcwd())
from models import ff_neuralnet, linear_classifier
import gui
from utils import sys_utils
from models import model_stage

"""
    Implementing interface between UI and state machines
"""

class Controller():
    def __init__(self):
        self.stages = []
        self.stages.append(model_stage.ModelStage())
        self.training_thread = TrainingThread()
        self.testing_thread = TesterThread()

    def dispatch_training_thread(self, stage_num, iterations):
        self.training_thread.set_stage(self.stages[stage_num], iterations)
        self.training_thread.start()

    def dispatch_testing_thread(self, stage_num):
        self.testing_thread.set_stage(self.stages[stage_num])
        self.testing_thread.start()

    def load_dataset(self, ds_name, num_batches, stage_num):
        did_load, msg = self.stages[stage_num].set_dataset(ds_name, num_batches)
        return did_load, msg
    
    def validate_ds(self, stage_num):
        return(self.stages[stage_num].dataset_isloaded())
    
    def get_dataset_info(self, stage_num):
        return(self.stages[stage_num].get_dataset_info())
    
    def clear_dataset(self, stage_num):
        self.stages[stage_num].clear_dataset()
    
    def validate_model(self, stage_num):
        return(self.stages[stage_num].model_isloaded())
    
    def initialize_model(self, 
                         model_type, layer_dims, stage_num, lrn_rate=.03, 
                         hidden_fn="tanh", output_fn="sigmoid", 
                         init_type="scalar"):
        loaded, msg = self.stages[stage_num].set_model(model_type, layer_dims, 
                        hidden_fn=hidden_fn, output_fn=output_fn, lrn_rate=lrn_rate, weight_init=init_type)
        return loaded, msg
    
    def get_model_info(self, stage_num):
        return(self.stages[stage_num].get_model_info())
    
    def clear_model(self, stage_num):
        self.stages[stage_num].clear_model()

    def save_model(self, stage_num, filename):
        self.stages[stage_num].save_model(filename)  

    def load_model(self, stage_num, filename):
        self.stages[stage_num].load_model(filename) 
    
    def get_stage_info(self, stage_num):
        return(self.stages[stage_num].get_stage_info())

    def add_stage(self):
        self.stages.append(model_stage.ModelStage())

    def delete_stage(self, stage_num):
        self.stages.pop(stage_num)

    def save_results(self, stage_num):
        self.stages[stage_num].save_results()
    

class TrainingThread(QThread):
    update_signal = pyqtSignal(int, float)
    finished_signal = pyqtSignal(bool, str)
    def __init__(self):
        super().__init__()
        self.stage = ""
        self.training_iterations = 0

    def set_stage(self, stage, train_iters):
        self.stage = stage
        self.training_iterations = train_iters

    def clear_stage(self):
        self.stage = ""
        self.training_iterations = 0        

    def run(self):
        did_finish, msg = self.stage.do_train(self.training_iterations, self.update_signal)
        self.finished_signal.emit(did_finish,msg)
        self.clear_stage()
        self.quit()

class TesterThread(QThread):
    finished_signal = pyqtSignal(bool,str)
    def __init__(self):
        super().__init__()
        self.frame = ""
    
    def set_stage(self, stage):
        self.stage = stage

    def clear_stage(self):
        self.stage = ""

    def run(self):
        did_test, acc = self.stage.do_test()
        self.finished_signal.emit(did_test, acc)
        self.clear_stage()
        self.quit()
