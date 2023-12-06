import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QGridLayout, QPushButton, QTextEdit, QProgressBar, QComboBox, QTabWidget, QTabBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import datetime
from PyQt5.QtCore import Qt, QThread
import os
import driver
from utils import sys_utils

class ModelTesterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.driver = driver.Driver()
        self.num_frames = (len(self.driver.model_frames))
        self.current_frame = 0
        self.worker_thread = QThread()
        self.initUI()

    def initUI(self):
    # Create layout
        current_row = 1
        now = datetime.datetime.now()
        self.session_name = "Session-{}".format(now.strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_buffer = []
        layout = QGridLayout(self)
        # Set dark mode background
        self.setStyleSheet('background-color: #2E2E2E; color: white;')

    # Create heading/title area
        title_label = QLabel('ML Model Tester', self)
        title_label.setStyleSheet('font-size: 36px; font-weight: bold; color: white;')
        layout.addWidget(title_label, current_row, 0, 2, 6)
        current_row = current_row + 2

        """
    # Section 1: Dataset
        """
        section1_label = QLabel('Dataset Selections', self)
        section1_label.setStyleSheet('font-size: 20px; font-weight: bold; color: white;')
        layout.addWidget(section1_label, current_row, 0, 1, 6)
        current_row = current_row + 1

        data_labels = ['Train/Dev/Test %:']
        self.data_input_fields = [QLineEdit(self) for _ in range(len(data_labels))]

    # data status window
        self.dataset_status_window = QTextEdit(self)
        self.dataset_status_window.setReadOnly(True)  # Make it read-only
        self.dataset_status_window.setStyleSheet('font-size: 14px; color: white;')
        layout.addWidget(self.dataset_status_window, current_row, 2, len(data_labels)+1, 2)

    # Add a dropdown selector for datasets
        dataset_selector_label = QLabel('Select Dataset:', self)
        dataset_selector_label.setStyleSheet('font-size: 20px; color: white;')
        layout.addWidget(dataset_selector_label, current_row, 0, 1, 1)

        self.dataset_selector = QComboBox(self)
        self.dataset_selector.setStyleSheet('font-size: 16px; color: white; border: 1px solid white;')
        available_datasets = sys_utils.get_available_datasets()
        self.dataset_selector.addItems(available_datasets)
        self.dataset_selector.currentIndexChanged.connect(self.dataset_selector_changed)
        layout.addWidget(self.dataset_selector, current_row, 1, 1, 1)
        current_row = current_row + 1

    # data input fields
        for i, (label, input_field) in enumerate(zip(data_labels, self.data_input_fields), start=current_row):
            label_widget = QLabel(label, self)
            label_widget.setStyleSheet('font-size: 20px; color: white;')  # Increase font size for labels
            layout.addWidget(label_widget, i, 0, 1, 1)
            layout.addWidget(input_field, i, 1, 1, 1)
            current_row = current_row + 1

    # data buttons
        self.load_dataset_button = QPushButton('Load Dataset', self)
        self.load_dataset_button.clicked.connect(self.load_dataset_button_click)
        self.load_dataset_button.setStyleSheet('font-size: 16px; font-weight: bold; background-color: grey; color: white;')
        layout.addWidget(self.load_dataset_button, current_row, 1)

        self.clear_dataset_button = QPushButton('Clear Dataset', self)
        self.clear_dataset_button.clicked.connect(self.clear_dataset_button_click)
        self.clear_dataset_button.setStyleSheet('font-size: 16px; font-weight: bold; background-color: grey; color: white;')
        layout.addWidget(self.clear_dataset_button, current_row, 2)
        current_row = current_row + 1

        """
    # Section 2: Model Parameters
        """
        section2_label = QLabel('Model Parameters', self)
        section2_label.setStyleSheet('font-size: 20px; font-weight: bold; color: white;')
        layout.addWidget(section2_label, current_row, 0)
        current_row = current_row + 1

    # model labels
        model_labels = ["Layer Dimensions:", 'Learning Rate:', 'Hidden Layer Activation', 
                        'Output Activation', 'Weight Initialization Type', 'Learning Type:']
        self.model_input_fields = [QLineEdit(self) for _ in range(len(model_labels))]

    # Add space for plots
        #self.plot_widget = PlotWidget(self)
        #layout.addWidget(self.plot_widget, current_row, 3,len(model_labels)+2,3)

    # Add model status window
        self.model_status_window = QTextEdit(self)
        self.model_status_window.setReadOnly(True)  # Make it read-only
        self.model_status_window.setStyleSheet('font-size: 14px; color: white;')
        layout.addWidget(self.model_status_window, current_row, 2, len(model_labels), 2)

    # Add a dropdown selector for models
        model_selector_label = QLabel('Select Model Type:', self)
        model_selector_label.setStyleSheet('font-size: 20px; color: white;')
        layout.addWidget(model_selector_label, current_row, 0, 1, 1)

        self.model_selector = QComboBox(self)
        self.model_selector.setStyleSheet('font-size: 16px; color: white; border: 1px solid white;')
        models = sys_utils.get_available_models()
        self.model_selector.addItems(models)
        self.model_selector.currentIndexChanged.connect(self.model_selector_changed)
        layout.addWidget(self.model_selector, current_row, 1, 1, 1)
        current_row = current_row + 1

    # add model labels
        for i, (label, input_field) in enumerate(zip(model_labels, self.model_input_fields), start=current_row):
            label_widget = QLabel(label, self)
            label_widget.setStyleSheet('font-size: 20px; color: white;')  # Increase font size for labels
            layout.addWidget(label_widget, i, 0)
            layout.addWidget(input_field, i, 1)
            current_row = current_row+1

    # model buttons
        self.initialize_model_button = QPushButton('Initialize Model', self)
        self.initialize_model_button.clicked.connect(self.initialize_model_button_click)
        self.initialize_model_button.setStyleSheet('font-size: 16px; font-weight: bold; background-color: grey; color: white;')
        layout.addWidget(self.initialize_model_button, current_row, 1)

        self.clear_model_button = QPushButton('Clear Model', self)
        self.clear_model_button.clicked.connect(self.clear_model_button_click)
        self.clear_model_button.setStyleSheet('font-size: 16px; font-weight: bold; background-color: grey; color: white;')
        layout.addWidget(self.clear_model_button, current_row, 2)
        current_row = current_row + 1

    # Create progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Training Progress")
        self.progress_bar.setStyleSheet('font-size: 12px; font-weight: bold; background-color: black; color: black;')
        layout.addWidget(self.progress_bar, current_row, 1, 1, 1)  # Adjusted row and column span as needed
        current_row = current_row + 1

    # Add Logging window
        self.log_window = QTextEdit(self)
        self.log_window.setReadOnly(True)  # Make it read-only
        layout.addWidget(self.log_window, current_row, 0, 1, 2)  # Adjusted row and column span as needed
        current_row = current_row + 1

    # Add Save Logs button
        self.save_logs_button = QPushButton('Save Logs', self)
        self.save_logs_button.clicked.connect(self.save_logs)
        self.save_logs_button.setStyleSheet('font-size: 16px; font-weight: bold; background-color: grey; color: white;')
        layout.addWidget(self.save_logs_button, current_row, 0,1,1)

        """
    # Add add frame button
        self.add_frame_button = QPushButton('Add Frame', self)
        self.add_frame_button.clicked.connect(self.add_frame)
        self.add_frame_button.setStyleSheet('font-size: 16px; font-weight: bold; background-color: grey; color: white;')
        layout.addWidget(self.add_frame_button, current_row, 1,1,1)

    # Add delete frame button
        self.delete_frame_button = QPushButton('Delete Frame', self)
        self.delete_frame_button.clicked.connect(self.delete_frame)
        self.delete_frame_button.setStyleSheet('font-size: 16px; font-weight: bold; background-color: grey; color: white;')
        layout.addWidget(self.delete_frame_button, current_row, 2,1,1)

    # Add switch frame button
        self.switch_frame_button = QPushButton('Switch Frame: {}'.format(self.current_frame), self)
        self.switch_frame_button.clicked.connect(self.switch_frame)
        self.switch_frame_button.setStyleSheet('font-size: 16px; font-weight: bold; background-color: grey; color: white;')
        layout.addWidget(self.switch_frame_button, current_row, 5,1,1)
        """

    # Set the main layout for the window
        self.setLayout(layout)

    # Set window properties with a larger default size
        self.setGeometry(0, 0, 1800, 900)  # Adjusted size with space for plots
        self.setWindowTitle('Qt Application with Input Fields and Plots')

    # Show the window
        self.show()

    def load_dataset_button_click(self):
        ds_name = self.dataset_selector.currentText()
        loaded, msg = self.driver.load_dataset(ds_name, self.current_frame)
        if(loaded):
            self.write_to_data_status_window(msg)
            self.log("Loaded Dataset: {}. See dataset info".format(ds_name), "DATA")
        self.log(msg, "DATA")

    def clear_dataset_button_click(self):
        self.driver.clear_dataset(self.current_frame)
        self.log("Clearing data from frame {}".format(self.current_frame), "DATA")
        self.write_to_data_status_window("")
        
    def initialize_model_button_click(self):
        if(self.driver.validate_model(self.current_frame)):
            if(self.driver.validate_ds(self.current_frame)):
                self.train_model_button_click()
                return
            else:
                self.log("No dataset loaded!", "APP")
                return
        ml_type = self.model_selector.currentText()
        if(ml_type == ""):
            self.log("Please specify a model type to initialize", "APP")
            return
        if(self.model_input_fields[0].text() == ""):
            self.log("Please provide a comma seperated list of integers for layer dimensions. " +
                     "*** Do not include feature vector size ***", "APP")
            return
        else:
            layer_dims = self.model_input_fields[0].text().split(",")
            lrn_rate, hidden_fn, output_fn, init_type = .03, "tanh", "sigmoid", "scalar"
            for i in range(len(layer_dims)):
                layer_dims[i] = int(layer_dims[i])
            if(self.model_input_fields[1].text() != ""):
                lrn_rate = float(self.model_input_fields[1].text())
            if(self.model_input_fields[2].text() != ""):
                hidden_fn = self.model_input_fields[2].text()
            if(self.model_input_fields[3].text() != ""):
                output_fn = self.model_input_fields[3].text()
            if(self.model_input_fields[4].text() != ""):
                init_type = self.model_input_fields[4].text()
            #(self,model_type, layer_dims, frame, lrn_rate=.03, hidden_fn="tanh", output_fn="sigmoid", init_type="scalar"):
            success, msg = self.driver.initialize_model(ml_type, layer_dims, self.current_frame, lrn_rate, hidden_fn, output_fn, init_type)
            if(success):
                self.initialize_model_button.setText("Train Model")
                self.write_to_model_status_window(msg)
            self.log(msg, "MODEL")

    def clear_model_button_click(self):
        self.driver.clear_model(self.current_frame)
        self.log("Clearing model from frame {}".format(self.current_frame), "MODEL")
        self.write_to_model_status_window("")
        self.initialize_model_button.setText("Initialize Model")

    def train_model_button_click(self):
        self.driver.train_model(self.current_frame)

    def log(self, message, tag="DEFAULT"):
        # Function to log messages to the QTextEdit
        now = datetime.datetime.now()
        message = "[{}] [{}] ".format(now.strftime("%m/%d/%Y-%H:%M:%S"), tag) + message
        self.log_window.append(message)
        if(message.count("Writing logs to") == 0):
            self.log_buffer.append(message)

    def add_frame(self):
        self.driver.add_frame()
        self.num_frames = self.num_frames + 1
        self.log("Adding empty model frame", "APP")
        self.update_switch_frame_button()

    def delete_frame(self):
        if(len(self.driver.model_frames) == 1):
            self.log("Must have at least one frame!", "APP")
            return
        self.driver.delete_frame(self.current_frame)
        self.switch_frame("back")
        self.update_switch_frame_button()

    def switch_frame(self, dir="fwd"):
        self.current_frame = self.current_frame + 1
        if(self.current_frame > len(self.driver.model_frames) - 1):
            self.current_frame = 0
        frame_info = self.driver.get_frame_info(self.current_frame)
        self.write_to_data_status(frame_info[0])
        if(dir == "back"):
            self.current_frame = self.current_frame -1
            if(self.current_frame < 0):
                self.current_frame = 0
        self.update_switch_frame_button()

    def save_logs(self):
        proj_dir = os.path.dirname(os.path.realpath(__file__))
        results_folder = proj_dir + "\\..\\results\\"
        results_file = results_folder + "logs\\" + self.session_name
        self.log("Writing logs to {}".format(results_file), "SYS")
        f = open(results_file, "a")
        for i in range(len(self.log_buffer)):
            f.write(str(self.log_buffer[i])+"\n")
        self.log_buffer.clear()
        f.close()

    def update_switch_frame_button(self):
        self.switch_frame_button.setText('Switch Frame: {}/{}'.format(self.current_frame, len(self.driver.model_frames)-1))

    def updateProgressBar(self):
        # Example function to update progress bar (just for demonstration)
        value = self.progress_bar.value() + 10
        if value > 100:
            value = 0
        self.progress_bar.setValue(value)

    def write_to_data_status_window(self, message):
        self.dataset_status_window.setText(message)

    def write_to_model_status_window(self, message):
        self.model_status_window.setText(message)

    def dataset_selector_changed(self):
        pass

    def model_selector_changed(self):
        pass
    
class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initUI()

    def initUI(self):
        # Create a FigureCanvas for displaying plots
        self.canvas = FigureCanvas(plt.Figure())
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModelTesterApp()
    
    window.log("Session Started: " + window.session_name, "APP")
    window.log("Available Datasets: " + str(sys_utils.get_available_datasets()), "SYS")
    window.log("Available Models: " + str(sys_utils.get_available_models()), "SYS")
    sys.exit(app.exec_())