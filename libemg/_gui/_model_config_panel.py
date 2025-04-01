import dearpygui.dearpygui as dpg
import os
import json
import libemg

from pathlib import Path
from PIL import Image as PILImage

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

import numpy as np
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter, FilePackager
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier, EMGRegressor, OnlineEMGRegressor
from libemg.environments.controllers import ClassifierController, RegressorController
import threading
from multiprocessing import Process, Queue, Manager
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from functools import partial
import time
from collections import deque
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import re

# Class made by me
class ModelConfigPanel:
    '''
    The Model Configuration Panel class. 
    Parameters
    ----------
    deadband: float, default=0.1
        The deadband value for the regression model. If value is less than deadband, the model will output 0. 
    alpha_mf1: int, default=20
        The threshold angle for motor function 1. Must be between 0 and 45 degrees.
    alpha_mf2: int, default=20
        The threshold angle for motor function 2. Must be between 0 and 45 degrees.
    gain_mf1: float, default=0.5
        The gain for motor function 1. Must be between 0.5 and 3.
    gain_mf2: float, default=0.5
        The gain for motor function 2. Must be between 0.5 and 3.
    window_size: int, default=150
        The window size for the model (in ms?). 
    window_increment: int, default=50
        The window increment for the model (in ms?). 
    gui: GUI
        The GUI object this panel is associated with.
    '''
    def __init__(self,
                 deadband=0.1,
                 alpha_mf1=20,
                 alpha_mf2=20,
                 gain_mf1=0.5,
                 gain_mf2=0.5,
                 window_size=150,
                 window_increment=50,
                 data_folder='./data/',
                 gui=None):
        self.deadband = deadband
        self.alpha_mf1 = alpha_mf1
        self.alpha_mf2 = alpha_mf2
        self.gain_mf1 = gain_mf1
        self.gain_mf2 = gain_mf2 
        self.window_size = window_size
        self.window_increment = window_increment
        self.gui = gui
        self.data_folder = data_folder
        
        manager = Manager()
        self.configuration = manager.dict({
            "__mc_deadband": self.deadband,
            "__mc_alpha_mf1": self.alpha_mf1,
            "__mc_alpha_mf2": self.alpha_mf2,
            "__mc_gain_mf1": self.gain_mf1,
            "__mc_gain_mf2": self.gain_mf2,
            "__mc_window_size": self.window_size,
            "__mc_window_increment": self.window_increment,
            "running": False
        })
        self.model = None
        self.prediction_queue = Queue()
        self.plot_process = None
        
        self.UPD_IP = "127.0.0.1"
        self.UDP_PORT = 5005

        self.widget_tags = {"configuration": ['__mc_configuration_window', '__mc_deadband', '__mc_alpha_mf1', '__mc_alpha_mf2', '__mc_gain_mf1', '__mc_gain_mf2', '__mc_window_size', '__mc_window_increment']}
               

    def cleanup_window(self, window_name): # Don't really know what this does
        widget_list = self.widget_tags[window_name]
        for w in widget_list:
            if dpg.does_alias_exist(w):
                dpg.delete_item(w)

    def spawn_configuration_window(self):
        self.cleanup_window("configuration")
        with dpg.window(tag="__mc_configuration_window",
                        label="Model Configuration",
                        width=900,
                        height=480):
            dpg.add_text(label="Model Configuration")

            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):

                dpg.add_table_column(label="")
                dpg.add_table_column(label="")
                dpg.add_table_column(label="")

                # Add window size and increment for predictor
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Window Size: ")
                        dpg.add_input_int(default_value=self.window_size, 
                                            tag="__mc_window_size",
                                            width=100, 
                                            callback=self.update_value_callback
                                        )                   

                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Window Increment: ")
                        dpg.add_input_int(default_value=self.window_increment,
                                            tag="__mc_window_increment",
                                            width=100,
                                            callback=self.update_value_callback # Give another callback, that gets settings, updates plot and updates model
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Reset Model", 
                                       width = 100,
                                       callback=self.reset_model_callback
                                    )
                        
                # Add deadband and threshold for regressor   
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Deadband (%): ")
                        dpg.add_input_float(default_value=self.deadband, 
                                            tag="__mc_deadband",
                                            width=100,
                                            min_value=0,
                                            max_value=0.4,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )  
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Threshold angle mf1 (degrees): ")
                        dpg.add_input_int(default_value=self.alpha_mf1,
                                            tag="__mc_alpha_mf1",
                                            width=100,
                                            min_value=0,
                                            max_value=45,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Threshold angle mf2 (degrees): ")
                        dpg.add_input_int(default_value=self.alpha_mf2,
                                            tag="__mc_alpha_mf2",
                                            width=100,
                                            min_value=0,
                                            max_value=45,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Gain mf1: ")
                        dpg.add_input_float(default_value=self.gain_mf1,
                                            tag="__mc_gain_mf1",
                                            width=100,
                                            min_value=0.5,
                                            max_value=3,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="Gain mf2: ")
                        dpg.add_input_float(default_value=self.gain_mf2,
                                            tag="__mc_gain_mf2",
                                            width=100,
                                            min_value=0.5,
                                            max_value=3,
                                            min_clamped=True,
                                            max_clamped=True,
                                            callback=self.update_value_callback
                                        )

                # Visualization buttons
                with dpg.table_row():
                    dpg.add_button(label="Visualize Prediction", callback=self.prediction_btn_callback)
                    dpg.add_button(label="Visualize Raw EMG", callback=self.plot_raw_data_callback)
                    #dpg.add_button(label="Save Configuration", callback=self.save_configuration)

    def update_value_callback(self, sender, app_data):
        self.configuration[sender] = app_data
        print(self.configuration)


    def reset_model_callback(self):
        #stop plotting with controller, reset controller or something
       #self.configuration["running"] = False
        self.stop_prediction_plot()
        self.stop_controller()
        if self.model is not None: self.model.stop_running()
        self.get_settings()
        self.set_up_model()
        self.model.run(block=False)
        ("Model reset. Press Visualize Prediction to start again.")
        self.spawn_configuration_window()


    def get_settings(self):
        self.deadband = float(dpg.get_value(item="__mc_deadband"))
        self.gain_mf1 = float(dpg.get_value(item="__mc_gain_mf1"))
        self.gain_mf2 = float(dpg.get_value(item="__mc_gain_mf2"))
        self.alpha_mf1 = int(dpg.get_value(item="__mc_alpha_mf1"))
        self.alpha_mf2 = int(dpg.get_value(item="__mc_alpha_mf2"))
        self.window_size = int(dpg.get_value(item="__mc_window_size"))
        self.window_increment = int(dpg.get_value(item="__mc_window_increment"))
        print("Settings updated")

    def stop_controller(self):
        self.controller = None
        if self.controller_thread and self.controller_thread.is_alive():
            self.controller_thread.join(timeout=1)
            print("Controller thread terminated")
            self.controller_thread = None

    def start_prediction_plot(self):
        if self.plot_process and self.plot_process.is_alive(): # Does this in the stop-function as well, so might be redundant
            self.stop_prediction_plot()

        self.configuration["running"] = True
        plotter = PredictionPlotter(self.gui.axis_images)
        self.plot_process = Process(target=plotter.run, args=(self.configuration,self.prediction_queue)) #self.plot_process = Process(target=self.plotter, args=(self.configuration, self.prediction_queue))
        self.plot_process.start()

    def _plot_helper(self, config, queue):
        plotter = PredictionPlotter()
        plotter.run(config, queue)

    def stop_prediction_plot(self):
        if self.plot_process and self.plot_process.is_alive():
            self.configuration["running"] = False
            self.plot_process.join(timeout=1)
            if self.plot_process.is_alive():
                self.plot_process.terminate()
                self.plot_process.join()
            print("Plot process terminated")
            self.plot_process = None

    def prediction_btn_callback(self):
        if not (self.gui.online_data_handler and sum(list(self.gui.online_data_handler.get_data()[1].values()))):
            raise ConnectionError('Attempted to start data collection, but data are not being received. Please ensure the OnlineDataHandler is receiving data.')
        
        if self.model is None:
            self.get_settings()
            self.set_up_model() # Could combine this with start_estimation, i.e. running in there
            #self.gui.online_data_handler.reset() # Litt usikker på om denne fungerer, eller om den trengs
            self.model.run(block=False)

        self.configuration["running"] = True
        
        self.run_controller()
        self.start_prediction_plot()       

        self.cleanup_window("configuration")
        self.spawn_configuration_window()
        

    def run_controller(self):
        self.controller_thread = threading.Thread(target=self._run_controller_helper)
        self.controller_thread.start()
        self.controller_thread.join(timeout=1)

    def _run_controller_helper(self):
        if self.gui.regression_selected:
            self.controller = RegressorController(ip=self.UPD_IP, port=self.UDP_PORT)
        else:
            self.controller = ClassifierController(output_format='predictions', ip=self.UPD_IP, port=self.UDP_PORT)
        
        while self.configuration["running"]:
            pred = self.controller.get_data(["predictions"])
            if pred is not None: 
                #print("In controller: ", pred)
                self.prediction_queue.put(pred)
                time.sleep(0.1) # Maybe not a good idea, but not to overload the system

        self.controller = None
                
   
    # @staticmethod
    # def plotter(config, pred_queue):
    #     # Initiliaze plot
    #     plt.style.use('ggplot')
    #     fig, ax = plt.subplots(figsize=(6,6)) # layout='constrained'
    #     #gs = gridspec.GridSpec(3, 3, width_ratios=[1, 3, 1], height_ratios=[1, 3, 1])   
    #     plt.subplots_adjust(bottom=0.35)  # Leave space for two sliders

    #     fig.suptitle('Live Regressor Output', fontsize=16)
    #     ax.set_xlabel('Prediction DOF 1') # Find a way to get the actual motor function
    #     ax.set_ylabel('Prediction DOF 2')
    #     #ax.set_xlim(-1.5, 1.5) # Added by me. To make the plot stay still. 
    #     #ax.set_ylim(-1.5, 1.5)
    #     ax.grid(True)
    #     # Plot coordinate axes
    #     ax.axhline(0, color='black', linewidth=1)
    #     ax.axvline(0, color='black', linewidth=1)

    #     # Create an deadband circle
    #     circle = plt.Circle((0, 0), config["__mc_deadband"], color='r', fill=False, linestyle='dashed')
    #     ax.add_patch(circle)

    #     tale_plot, = ax.plot([], [], 'o', color='gray', markersize=4, alpha=0.5, label='Tale') # My code
    #     current_plot, = ax.plot([], [], 'o', color='red', markersize=8, markeredgecolor='black', label='Current Prediction') # My code
    #     threshold_lines = [
    #         ax.plot([], [], 'b')[0],  # Line 1 (mirrored around x-axis)
    #         ax.plot([], [], 'b')[0],  # Line 2 (mirrored around x-axis)
    #         ax.plot([], [], 'b')[0],  # Line 3 (mirrored around y-axis)
    #         ax.plot([], [], 'b')[0]   # Line 4 (mirrored around y-axis)
    #     ]
    #     history = deque(maxlen=1000) # Queue to store history of predictions for the deadband
        
    #     def _calculate_range():
    #             # Convert deque to NumPy array for easy min/max calculations
    #             history_array = np.array(history)  # Shape (N, 2)

    #             x_min, x_max = np.min(history_array[:, 0]), np.max(history_array[:, 0])
    #             y_min, y_max = np.min(history_array[:, 1]), np.max(history_array[:, 1])

    #             x_range = x_max - x_min if x_max > x_min else 1
    #             y_range = y_max - y_min if y_max > y_min else 1

    #             max_range = max(x_range, y_range)
    #             return max_range
        
    #     def add_figures/():
    #         """ Hard coded for now. Find path automatically later, as well as a way to handle all figures/ """
    #         img_paths = [
    #             "./figures/gestures/hand_close.png",  # Top
    #             "./figures/gestures/hand_open.png",   # Bottom
    #             "./figures/gestures/pronation.png",   # Left
    #             "./figures/gestures/supination.png"   # Right
    #         ]
    #         positions = [
    #             (0.5, 1.2),   # Top
    #             (0.5, -0.3),  # Bottom
    #             (-0.3, 0.5),  # Left
    #             (1.3, 0.5)    # Right
    #         ]
    #         for img_path, (x, y) in zip(img_paths, positions):
    #             img = mpimg.imread(img_path)
    #             imagebox = OffsetImage(img, zoom=0.3)
    #             ab = AnnotationBbox(imagebox, (x, y), frameon=False, xycoords='axes fraction')
    #             ax.add_artist(ab)

    #         # for ax in ax_figures/:
    #         #     ax.set_xticks([])
    #         #     ax.set_yticks([])
    #         #     ax.set_frame_on(False)
            
    #         # for ax, img_path in zip(ax_figures/, ['./figures/gestures/hand_close.png', './figures/gestures/hand_open.png', './figures/gestures/pronation.png', './figures/gestures/supination.png']):
    #         #     img = plt.imread(img_path)
    #         #     #ax.imshow(img)
    #         #     imagebox = OffsetImage(img, zoom=0.1)
    #         #     ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
    #         #     ax.add_artist(ab)
    #         #     ax.axis('off')

        
            
    #     def _draw_threshold():
    #         thresh_rad = np.deg2rad(config["__mc_alpha_mf1"])  # Convert threshold angle to radians
    #         #Define endpoints for each set of lines (assuming symmetry around axes)
    #         x_vals = np.array([-1.5,1.5])
    #         y_vals1 = np.tan(thresh_rad) * x_vals  # Lines around x-axis
    #         y_vals2 = -y_vals1
    #         #ax.plot(x_vals, y_vals, 'r', label=f'±{thresh_rad}° to x-axis')

    #         thresh_rad = np.deg2rad((config["__mc_alpha_mf2"]))  # Convert threshold angle to radians
    #         y_vals = np.array([-1.5,1.5])
    #         x_vals1 = np.tan(thresh_rad) * y_vals  # Lines around y-axis
    #         x_vals2 = -x_vals1

    #         threshold_lines[0].set_data(x_vals, y_vals1)
    #         threshold_lines[1].set_data(x_vals, y_vals2)
    #         threshold_lines[2].set_data(x_vals1, y_vals)
    #         threshold_lines[3].set_data(x_vals2, y_vals)
        
    #     def _draw_deadband():
    #         if len(history) > 0:
    #             max_range = _calculate_range()    
    #             circle.set_radius(config["__mc_deadband"]*max_range)

    #     def _draw_prediction(tale):
    #         if not pred_queue.empty():
    #             pred = pred_queue.get() 
    #             print("In plotter: ", pred)
    #             pred[0] = pred[0] * config["__mc_gain_mf1"]
    #             pred[1] = pred[1] * config["__mc_gain_mf2"]
    #             history.append(pred)
    #             tale.append(pred)
    #             tale = tale[-5:] 
    #             tale_array = np.array(tale)
    #             if tale_array.shape[0] > 1:
    #                 tale_plot.set_xdata(tale_array[:, 0])
    #                 tale_plot.set_ydata(tale_array[:, 1])
                
    #             current_plot.set_xdata(np.array(tale)[-1:,0])
    #             current_plot.set_ydata(np.array(tale)[-1:,1])

    #         return tale_plot, current_plot
            
    #     def update(frame, tale):
    #         if not config["running"]: # Stop when running flag is False
    #             print("Stopping animation...")
    #             anim.event_source.stop() # Stop FuncAnimation
    #             plt.close(fig) # Close figure
    #             return
            
    #         _draw_threshold()
    #         _draw_deadband()
    #         tale_plot, current_plot = _draw_prediction(tale)
    #         fig.canvas.draw_idle()
    #         ax.relim()
    #         ax.autoscale_view()
    #         return tale_plot, current_plot
        
    #     add_figures/()        
    #     anim = FuncAnimation(fig, partial(update, tale=[], ), interval=50, blit=False, cache_frame_data=False, repeat=False)  # must return value or animation won't work
    #     plt.show()


    
    def set_up_model(self):
        # Step 1: Parse offline training data
        with open(self.data_folder + '/collection_details.json', 'r') as f:
            collection_details = json.load(f)
        
        def _match_metadata_to_data(metadata_file: str, data_file: str, class_map: dict) -> bool:
            """
            Ensures the correct animation metadata file is matched with the correct EMG data file.

            Args:
                metadata_file (str): Metadata file path (e.g., "animation/collection_hand_open_close.txt").
                data_file (str): EMG data file path (e.g., "data/regression/C_0_R_01_emg.csv").
                class_map (dict): Dictionary mapping class index (str) to motion filenames.

            Returns:
                bool: True if the metadata file corresponds to the class of the data file.
            """
            # Extract class index from data filename (C_{k}_R pattern)
            match = re.search(r"C_(\d+)_R", data_file)
            if not match:
                return False  # No valid class index found

            class_index = match.group(1)  # Extract class index as a string

            # Find the expected metadata file from class_map
            expected_metadata = class_map.get(class_index)
            if not expected_metadata:
                return False  # No matching motion found

            # Construct the expected metadata filename
            expected_metadata_file = f"animation/{expected_metadata}.txt"

            return metadata_file == expected_metadata_file
        
        num_motions = collection_details['num_motions']
        num_reps = collection_details['num_reps']
        motion_names = collection_details['classes']
        class_map = collection_details['class_map']
        
        if self.gui.regression_selected:
            regex_filters = [
                RegexFilter(left_bound = "regression/C_", right_bound="_R", values = [str(i) for i in range(num_motions)], description='classes'),
                RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = [str(i) for i in range(num_reps)], description='reps')
            ]
            metadata_fetchers = [
                FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=motion_names, description='labels'), package_function=lambda meta, data: _match_metadata_to_data(meta, data, class_map) ) #package_function=lambda x, y: True)
            ]
            labels_key = 'labels'
            metadata_operations = {'labels': 'last_sample'}
        else:
            metadata_fetchers = None
            labels_key = 'classes'
            metadata_operations = None
        # if self.gui.regression_selected:
        #     regex_filters = [
        #         RegexFilter(left_bound='regression/C_0_R_', right_bound='_emg.csv', values=['0'], description='reps') # reps are hard-coded, find a way to make it dynamic
        #     ]
        #     metadata_fetchers = [
        #         FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=['collection'], description='labels'), package_function=lambda x, y: True)
        #     ]
        #     labels_key = 'labels'
        #     metadata_operations = {'labels': 'last_sample'}
        # else:
        #     regex_filters = [
        #         RegexFilter(left_bound = "classification/C_", right_bound="_R", values = ["0","1","2","3","4"], description='classes'),
        #         RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = ["0", "1", "2"], description='reps'),
        #     ]
        #     metadata_fetchers = None
        #     labels_key = 'classes'
        #     metadata_operations = None

        offline_dh = OfflineDataHandler()
        offline_dh.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",")
        train_windows, train_metadata = offline_dh.parse_windows(self.window_size, self.window_increment, metadata_operations=metadata_operations)

        # Step 2: Extract features from offline data
        fe = FeatureExtractor()
        print("Extracting features")
        feature_list = fe.get_feature_groups()['HTD'] # Make this chosen from the GUI later
        training_features = fe.extract_features(feature_list, train_windows, array=True)

        # Step 3: Dataset creation
        data_set = {}
        print("Creating dataset")
        data_set['training_features'] = training_features
        data_set['training_labels'] = train_metadata[labels_key]

        # Step 4: Create the EMG model
        
        self.gui.online_data_handler.prepare_smm()
        model = self.gui.model_str
        print('Fitting model...')
        if self.gui.regression_selected:
            # Regression
            emg_model = EMGRegressor(model=model)
            emg_model.fit(feature_dictionary=data_set)
            # consider adding a threshold angle here, or just do this when setting up the controller
            emg_model.add_deadband(self.deadband) # Add a deadband to the regression model. Value below this threshold will be considered 0.
            self.model = OnlineEMGRegressor(emg_model, self.window_size, self.window_increment, self.gui.online_data_handler, feature_list, ip=self.UPD_IP, port=self.UDP_PORT)
        else:
            # Classification
            emg_model = EMGClassifier(model=model)
            emg_model.fit(feature_dictionary=data_set)
            emg_model.add_velocity(train_windows, train_metadata[labels_key])
            self.model = OnlineEMGClassifier(emg_model, self.window_size, self.window_increment, self.gui.online_data_handler, feature_list, output_format='probabilities', ip=self.UPD_IP, port=self.UDP_PORT)

        # Step 5: Create online EMG model and start predicting.
        print('Model fitted and running!')
        #self.model.run(block=False) # block set to false so it will run in a seperate process.
    
    
    ## Callbacks for plotting raw EMG - from LibEMG
    def plot_raw_data_callback(self):
        self.visualization_thread = threading.Thread(target=self._run_visualization_helper)
        self.visualization_thread.start()
 
    def _run_visualization_helper(self):
        self.gui.online_data_handler.visualize(block=False)




class PredictionPlotter:
    def __init__(self, 
                 axis_images={ 
            'N': PILImage.open(Path('images/gestures', 'hand_open.png')),
            'S': PILImage.open(Path('images/gestures', 'hand_close.png')),
            'E': PILImage.open(Path('images/gestures', 'pronation.png')),
            'W': PILImage.open(Path('images/gestures', 'supination.png'))
        }):
        # self.config = config
        # self.pred_queue = pred_queue
        self.history = deque(maxlen=1000)
        self.tale = []
        self.axis_images = axis_images
               
    def _initialize_plot(self, config):
        self.fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(3, 3, figure=self.fig, width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])

        # Create subplots in the grid
        self.ax_main = self.fig.add_subplot(gs[1, 1])   # Main (center)
        self.ax_north = self.fig.add_subplot(gs[0, 1])  # North (top center)
        self.ax_west = self.fig.add_subplot(gs[1, 0])   # West (left center)
        self.ax_east = self.fig.add_subplot(gs[1, 2])   # East (right center)
        self.ax_south = self.fig.add_subplot(gs[2, 1])  # South (bottom center)

        self.ax_main.set_title("Estimated Motor Function")
        self.ax_main.set_xlabel("MF 1")
        self.ax_main.set_ylabel("MF 2")
        self.ax_main.grid(True)
        #self.ax_main.plot(np.sin(np.linspace(0, 10, 100)))  # Example data
        # Plot of predictions
        self.tale_plot, = self.ax_main.plot([], [], 'o', color='gray', markersize=4, alpha=0.5, label='Tale')
        self.current_plot, = self.ax_main.plot([], [], 'o', color='red', markersize=8, markeredgecolor='black', label='Current Prediction')
        
        # Create a circle for the deadband
        self.circle = plt.Circle((0, 0), config["__mc_deadband"], color='r', fill=False, linestyle='dashed')
        self.ax_main.add_patch(self.circle)
        # Threshold lines
        self.threshold_lines = [
            self.ax_main.plot([], [], 'b')[0],
            self.ax_main.plot([], [], 'b')[0],
            self.ax_main.plot([], [], 'b')[0],
            self.ax_main.plot([], [], 'b')[0]
        ]

        # Figures for images of motor functions
        #self.ax_north.set_title()
        self.ax_north.imshow(self.axis_images['N'])

        #self.ax_south.set_title("South")
        self.ax_south.imshow(self.axis_images['S'])

        #self.ax_west.set_title("West")
        self.ax_west.imshow(self.axis_images['W'])

        #self.ax_east.set_title("East")
        self.ax_east.imshow(self.axis_images['E'])

        # Hide axis labels for surrounding figures
        for ax in [self.ax_north, self.ax_south, self.ax_west, self.ax_east]:
            ax.set_xticks([])
            ax.set_yticks([])


    def _calculate_range(self):
        if len(self.history) > 0:
            history_array = np.array(self.history)
            x_min, x_max = np.min(history_array[:, 0]), np.max(history_array[:, 0])
            y_min, y_max = np.min(history_array[:, 1]), np.max(history_array[:, 1])
            return max(x_max - x_min, y_max - y_min, 1)
        return 1
    
    def _draw_threshold(self, config):
        thresh_rad_x = np.deg2rad(config["__mc_alpha_mf1"])
        x_vals = np.array([-1.5, 1.5])
        y_vals1, y_vals2 = np.tan(thresh_rad_x) * x_vals, -np.tan(thresh_rad_x) * x_vals
        
        thresh_rad_y = np.deg2rad(config["__mc_alpha_mf2"])
        y_vals = np.array([-1.5, 1.5])
        x_vals1, x_vals2 = np.tan(thresh_rad_y) * y_vals, -np.tan(thresh_rad_y) * y_vals
        
        self.threshold_lines[0].set_data(x_vals, y_vals1)
        self.threshold_lines[1].set_data(x_vals, y_vals2)
        self.threshold_lines[2].set_data(x_vals1, y_vals)
        self.threshold_lines[3].set_data(x_vals2, y_vals)
    
    def _draw_deadband(self, config):
        self.circle.set_radius(config["__mc_deadband"] * self._calculate_range())
    
    def _draw_prediction(self, config, pred_queue):
        if not pred_queue.empty():
            pred = pred_queue.get()
            pred[0] *= config["__mc_gain_mf1"]
            pred[1] *= config["__mc_gain_mf2"]
            self.history.append(pred)
            self.tale.append(pred)
            self.tale = self.tale[-5:]
            tale_array = np.array(self.tale)
            
            if tale_array.shape[0] > 1:
                self.tale_plot.set_xdata(tale_array[:, 0])
                self.tale_plot.set_ydata(tale_array[:, 1])
            
            self.current_plot.set_xdata(tale_array[-1:, 0])
            self.current_plot.set_ydata(tale_array[-1:, 1])
    
    def update(self, frame, config, pred_queue):
        if not config["running"]:
            print("Stopping animation...")
            self.anim.event_source.stop()
            plt.close(self.fig)
            return
        
        self._draw_threshold(config=config)
        self._draw_deadband(config=config)
        self._draw_prediction(config=config, pred_queue=pred_queue)
        self.fig.canvas.draw_idle()
        self.ax_main.relim()
        self.ax_main.autoscale_view() # self.ax updatet initilize function27.03 11:52
    
    def run(self, config, pred_queue):
        #self._initialize_plot(config)
        self._initialize_plot(config)
        #self._add_images()
        self.anim = FuncAnimation(self.fig, partial(self.update, config=config, pred_queue=pred_queue), interval=50, blit=False, cache_frame_data=False, repeat=False)
        plt.tight_layout()
        plt.show()
    
    ########## Got from LibEMG (CartesianPlotAnimator in animator.py)  #################  
    def _format_figure(self):
        max_range = self._calculate_range()
        axis_limits = (-max_range*2, max_range*2)
        if self.axis_images is not None:
            #self.ax.axis('off')  # hide default axis
            # Make 3 x 3 grid
            grid_shape = (3, 3)
            gs = self.fig.add_gridspec(grid_shape[0], grid_shape[1], width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])

            # Create subplots using the gridspec
            axs = np.empty(shape=grid_shape, dtype=object)
            for row_idx in range(grid_shape[0]):
                for col_idx in range(grid_shape[1]):
                    ax = plt.subplot(gs[row_idx, col_idx])
                    if (row_idx, col_idx) != (1, 1):
                        # Disable axis for figures/, not for main plot
                        ax.axis('off')
                    axs[row_idx, col_idx] = ax

            loc_axis_map = {
                'NW': axs[0, 0],
                'N': axs[0, 1],
                'NE': axs[0, 2],
                'W': axs[1, 0],
                'E': axs[1, 2],
                'SW': axs[2, 0],
                'S': axs[2, 1],
                'SE': axs[2, 2]
            }
            for loc, image in self.axis_images.items():
                self.ax = loc_axis_map[loc]
                self.ax.imshow(image)
            # Set main axis so icon is drawn correctly
            plt.sca(axs[1, 1])    
            self.ax = axs[1, 1]
        
        ticks = [-1., -0.5, 0, 0.5, 1.]
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.axis('equal')
        self.ax.set(xlim=axis_limits, ylim=axis_limits)
        #return fig, ax
    
    @staticmethod
    def _add_image_label_axes(fig: Figure):
        """Add axes to a matplotlib Figure for displaying figures/ in the top, right, bottom, and left of the Figure. 
        
        Parameters
        ----------
        fig: matplotlib.pyplot.Figure
            Figure to add image axes to.
        
        Returns
        ----------
        np.ndarray
            Array of matplotlib axes objects. The location in the array corresponds to the location of the axis in the figure.
        """
        # Make 3 x 3 grid
        grid_shape = (3, 3)
        gs = fig.add_gridspec(grid_shape[0], grid_shape[1], width_ratios=[1, 2, 1], height_ratios=[1, 2, 1])

        # Create subplots using the gridspec
        axs = np.empty(shape=grid_shape, dtype=object)
        for row_idx in range(grid_shape[0]):
            for col_idx in range(grid_shape[1]):
                ax = plt.subplot(gs[row_idx, col_idx])
                if (row_idx, col_idx) != (1, 1):
                    # Disable axis for figures/, not for main plot
                    ax.axis('off')
                axs[row_idx, col_idx] = ax
        
        return axs