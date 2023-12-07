import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QGridLayout, QPushButton, QTextEdit, QProgressBar, QComboBox, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
import pyqtgraph
import datetime
from PyQt5.QtCore import Qt
import os
import driver
from utils import sys_utils
from config import UI_CONFIG

class ModelTesterUI(QWidget):
    def __init__(self):
        super().__init__()

        self.driver = driver.Driver()

        self.setGeometry(UI_CONFIG.STRT, UI_CONFIG.END, UI_CONFIG.LEN, UI_CONFIG.WID)
        self.setWindowTitle(UI_CONFIG.APP_TITLE)
        self.session_name = "Session-{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.num_frames = (len(self.driver.model_frames))
        self.current_frame = 0
        self.log_buffer = []

        self.initUI()

    def initUI(self):
        layout = QGridLayout(self)
        self.setStyleSheet('background-color: #2E2E2E; color: white;')

        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet('background-color: #2E2E2E; color: #2E2E2E;')
        tab_1 = QWidget()
        tab_2 = QWidget()
        self.tabs.addTab(tab_1, "Training Tab")
        self.tabs.addTab(tab_2, "Dev Tab")

        self.init_tab1(tab_1)
        self.init_tab2(tab_2)

        layout.addWidget(self.tabs)
        self.setLayout(layout)

        self.show()

    def init_tab1(self, tab_widget):
        current_row = 0
        layout = QGridLayout(tab_widget)
        current_row, layout = self.set_title_section(layout, current_row)
        current_row, layout = self.set_dataentry_section(layout, current_row)
        current_row, layout = self.set_model_parameters_section(layout, current_row)
        current_row, layout = self.set_logger_section(layout, current_row)
        current_row, layout = self.set_frame_buttons_section(layout, current_row)

        tab_widget.setLayout(layout)

    def init_tab2(self, tab_widget):
        layout = QGridLayout(tab_widget)
        current_row = 0
        current_row, layout = self.set_title_section(layout, current_row)

        self.cost_plot = pyqtgraph.PlotWidget()
        layout.addWidget(self.cost_plot, current_row, 0, 3, 3)
        pen = pyqtgraph.mkPen(color=(255, 255, 255), width=3, size="20pt")
        self.cost_data_buffer = []
        self.cost_plot.setTitle("Cost", color = "w")
        self.cost_plot.setLabel("left", "Cost")
        self.cost_plot.setLabel("bottom", "Training Iteration")
        self.cost_plot.showGrid(x=True, y=True)
        self.cost_plot.setXRange(0, 200)
        self.cost_plot.plotItem.setMouseEnabled(x=False)

        self.second_plot = pyqtgraph.PlotWidget()
        layout.addWidget(self.second_plot, current_row, 3, 3, 3)
        #pen = pyqtgraph.mkPen(color=(255, 255, 255), width=3, size="20pt")
        #self.cost_data_buffer = []
        self.second_plot.setTitle("Learn Rate", color = "w")
        self.second_plot.setLabel("left", "Cost")
        self.second_plot.setLabel("bottom", "Training Iteration")
        self.second_plot.showGrid(x=True, y=True)
        self.second_plot.setYRange(-1, 10)
        self.second_plot.plotItem.setMouseEnabled(y=False)

        self.third_plot = pyqtgraph.PlotWidget()
        layout.addWidget(self.third_plot, current_row+3, 0, 3, 3)
        #pen = pyqtgraph.mkPen(color=(255, 255, 255), width=3, size="20pt")
        #self.cost_data_buffer = []
        self.third_plot.setTitle("Cost", color = "w")
        self.third_plot.setLabel("left", "Cost")
        self.third_plot.setLabel("bottom", "Training Iteration")
        self.third_plot.showGrid(x=True, y=True)
        self.third_plot.setYRange(-1, 10)
        self.third_plot.plotItem.setMouseEnabled(y=False)

        self.fourth_plot = pyqtgraph.PlotWidget()
        layout.addWidget(self.fourth_plot, current_row+3, 3, 3, 3)
        #pen = pyqtgraph.mkPen(color=(255, 255, 255), width=3, size="20pt")
        #self.cost_data_buffer = []
        self.fourth_plot.setTitle("Cost", color = "w")
        self.fourth_plot.setLabel("left", "Cost")
        self.fourth_plot.setLabel("bottom", "Training Iteration")
        self.fourth_plot.showGrid(x=True, y=True)
        self.fourth_plot.setYRange(-1, 10)
        self.fourth_plot.plotItem.setMouseEnabled(y=False)

        tab_widget.setLayout(layout)

    def update_plot(self, cost):
        self.cost_data_buffer.append(cost)
        self.cost_plot.plot(self.cost_data_buffer)

    def set_title_section(self, layout, current_row):
        title_label = QLabel('ML Model Tester', self)
        title_label.setStyleSheet('font-size: 36px; font-weight: bold; color: white;')
        layout.addWidget(title_label, current_row, 0, 1, 6)
        current_row = current_row + 1
        return(current_row, layout)

    def set_dataentry_section(self, layout, current_row):
        section1_label = QLabel('Dataset Selections', self)
        section1_label.setStyleSheet('font-size: 20px; font-weight: bold; color: white;')
        layout.addWidget(section1_label, current_row, 0, 0, 6)
        current_row = current_row + 1
        self.data_labels = ['Number of MiniBatches:']
        default_values = [1]
        self.data_input_fields = [QLineEdit(self) for _ in range(len(self.data_labels))]

        # data status window
        self.dataset_status_window = QTextEdit(self)
        self.dataset_status_window.setReadOnly(True)  # Make it read-only
        self.dataset_status_window.setStyleSheet('font-size: 14px; color: white;')
        layout.addWidget(self.dataset_status_window, current_row, 2, len(self.data_labels)+1, 2)

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
        for i, (label, input_field) in enumerate(zip(self.data_labels, self.data_input_fields), start=0):
            label_widget = QLabel(label, self)
            label_widget.setStyleSheet('font-size: 20px; color: white;')  # Increase font size for labels
            input_field.setText(str(default_values[i]))
            input_field.setStyleSheet('font-size: 16px; color: white;')
            layout.addWidget(label_widget, current_row, 0, 1, 1)
            layout.addWidget(input_field, current_row, 1, 1, 1)
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

        return(current_row, layout)
    
    def set_model_parameters_section(self, layout, current_row):
        section2_label = QLabel('Model Parameters', self)
        section2_label.setStyleSheet('font-size: 20px; font-weight: bold; color: white;')
        layout.addWidget(section2_label, current_row, 0)
        current_row = current_row + 1

    # model labels
        self.model_labels = ["Layer Dimensions:", 'Learning Rate:', 'Hidden Layer Activation', 
                        'Output Activation', 'Weight Initialization Type', 'Training Iterations:']
        self.model_input_fields = [QLineEdit(self) for _ in range(len(self.model_labels))]
        default_values = ["5,5,1", .03, "tanh", "sigmoid", "scalar", "3000"]

    # Add model status window
        self.model_status_window = QTextEdit(self)
        self.model_status_window.setReadOnly(True)  # Make it read-only
        self.model_status_window.setStyleSheet('font-size: 14px; color: white;')
        layout.addWidget(self.model_status_window, current_row, 2, len(self.model_labels)+1, 2)

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
        for i, (label, input_field) in enumerate(zip(self.model_labels, self.model_input_fields), start=0):
            label_widget = QLabel(label, self)
            label_widget.setStyleSheet('font-size: 20px; color: white;')  # Increase font size for labels
            input_field.setText(str(default_values[i]))
            input_field.setStyleSheet('font-size: 16px; color: white;')
            layout.addWidget(label_widget, current_row, 0)
            layout.addWidget(input_field, current_row, 1)
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
        
        return(current_row, layout)
    
    def set_logger_section(self, layout, current_row):
        # Add Logging window
        self.log_window = QTextEdit(self)
        self.log_window.setReadOnly(True)  # Make it read-only
        self.log_window.setStyleSheet('font-size: 16px; background-color: black; color: white;')
        layout.addWidget(self.log_window, current_row, 0, 1, 2)  # Adjusted row and column span as needed
        current_row = current_row + 1

        # Add Save Logs button
        self.save_logs_button = QPushButton('Save Logs', self)
        self.save_logs_button.clicked.connect(self.save_logs)
        self.save_logs_button.setStyleSheet('font-size: 16px; font-weight: bold; background-color: grey; color: white;')
        layout.addWidget(self.save_logs_button, current_row, 0,1,1)
        return(current_row, layout)
    
    def set_frame_buttons_section(self, layout, current_row):
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
        return(current_row, layout)

    def set_plotter_section(self, layout, current_row, plot_labels):
        return(current_row, layout)
    
    def get_ds_params(self):
        num_batches_input = self.data_input_fields[0].text()
        ds_name = self.dataset_selector.currentText()
        params = [[ds_name, True]]
        if(num_batches_input ==""):
            param = ["NONE", True]
        else:
            try:
                batch_size = int(self.data_input_fields[0].text())
            except ValueError:
                self.log("Number of Minibatches must be an integer")
                return
            param = [batch_size, True]
        params.append(param)
        return params
    
    def load_dataset_button_click(self):
        params = self.get_ds_params()
        for i in range(len(params)):
            if(params[0] == "NONE" and params[1] == True):
                self.log("No {} provided!".format(self.data_labels[i]), "DATA")
                return
        loaded, msg = self.driver.load_dataset(params[0][0], params[0][1], self.current_frame)
        if(loaded):
            self.write_to_data_status_window(msg)
            self.log("Loaded Dataset: {}. See dataset info".format(params[0][0]), "DATA")
        self.log(msg, "DATA")

    def clear_dataset_button_click(self):
        self.driver.clear_dataset(self.current_frame)
        self.log("Clearing data from frame {}".format(self.current_frame), "DATA")
        self.write_to_data_status_window("")

    def get_model_params(self):
        model_type = self.model_selector.currentText()
        params = [[model_type, True]]
        layer_dims = self.model_input_fields[0].text().split(",")
        for i in range(len(layer_dims)):
            try:
                layer_dims[i] = int(layer_dims[i])
            except ValueError:
                self.log("All layer dimensions must be integers", "MODEL")
                return params, False
        params.append([layer_dims, True])
        try:
            lrn_rate = float(self.model_input_fields[1].text())
        except ValueError:
            self.log("Learning Rate must be a float", "MODEL")
            return params, False
        params.append([lrn_rate, True])
        hidden_act_fn = self.model_input_fields[2].text()
        params.append([hidden_act_fn, True])
        output_act_fn = self.model_input_fields[3].text()
        params.append([output_act_fn, True])
        weight_init_type = self.model_input_fields[4].text()
        params.append([weight_init_type, True])
        try:
            iterations = int(self.model_input_fields[5].text())
        except ValueError:
            self.log("Training Iterations must be int", "MODEL")
            return params, False
        params.append([iterations, True])
        return params, True
        
    def initialize_model_button_click(self):
        if(self.driver.validate_model(self.current_frame)):
            if(self.driver.validate_ds(self.current_frame)):
                self.train_model_button_click()
                return
            else:
                self.log("No dataset loaded!", "APP")
                return
        params, params_success = self.get_model_params()
        if(params_success):
            init_success, msg = self.driver.initialize_model(params[0][0], params[1][0], self.current_frame, params[2][0], params[3][0], params[4][0], params[5][0], params[6][0])
            if(init_success):
                self.initialize_model_button.setText("Train Model")
                self.write_to_model_status_window(msg)
            self.log(msg, "MODEL")

    def clear_model_button_click(self):
        self.driver.clear_model(self.current_frame)
        self.log("Clearing model from frame {}".format(self.current_frame), "MODEL")
        self.write_to_model_status_window("")
        self.initialize_model_button.setText("Initialize Model")

    def train_model_button_click(self):
        self.driver.signal.connect(self.training_updated)
        self.driver.start()
        #msg = self.driver.train_model(self.current_frame)
        #self.plot_cost(msg)

    def training_updated(self, iter, cost):
        self.update_progress_bar(iter)
        self.update_plot(cost)

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

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def write_to_data_status_window(self, message):
        self.dataset_status_window.setText(message)

    def write_to_model_status_window(self, message):
        self.model_status_window.setText(message)

    def dataset_selector_changed(self):
        pass

    def model_selector_changed(self):
        pass

    def plot_cost(self, data):
        pass  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModelTesterUI()
    
    window.log("Session Started: " + window.session_name, "APP")
    window.log("Available Datasets: " + str(sys_utils.get_available_datasets()), "DATA")
    window.log("Available Models: " + str(sys_utils.get_available_models()), "MODEL")
    sys.exit(app.exec_())