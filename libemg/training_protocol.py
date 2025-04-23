from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import cv2

from tkinter import *
from libemg.streamers import delsys_streamer 
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter, FilePackager
from libemg.animator import ScatterPlotAnimator, ArrowPlotAnimator, MediaGridAnimator
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier, EMGRegressor, OnlineEMGRegressor
from libemg.environments.controllers import ClassifierController, RegressorController
from model_gui import ML_GUI
from libemg.environments.fitts import ISOFitts, FittsConfig
import re, json

class TrainingProtocol:
    """Class to handle the training protocol for the EMG classifier/regressor. Inspired by the Menu class in Menu.py.
    Note! Only regression is supported for now.
    """
    def __init__(self):
        streamer, sm = delsys_streamer(channel_list=[0,1,2,3,8]) # returns streamer and the shared memory object, need to give in the active channels number -1, so 2 is sensor 3
        # Create online data handler to listen for the data
        self.odh = OnlineDataHandler(sm)
        # Learning model
        self.model = None
        self.model_str = None

        #self.training_data_folder = Path('data', 'classification').absolute().as_posix() # Folder where training data is stored. Used in set_up_model and ML_GUI, hence here for consistency. Changed in regression_selected() to regression data folder. Nvm, that doesn't work

        # Initialize motor functions here for easier access
        self.motor_functions = {
            'hand_open_close': (1, 0),          # Movement along x-axis
            'pronation_supination': (0, 1),     # Movement along y-axis
            'diagonal_1': (1, 1)*1/np.sqrt(2),               # Diagonal movement (↗)
            'diagonal_2': (1, -1)*1/np.sqrt(2),              # Diagonal movement (↘)
        }
        # TODO: Add images the simultanous gestures
        self.axis_media = {
            'N': PILImage.open(Path('images/gestures', 'pronation.png')),
            'S': PILImage.open(Path('images/gestures', 'supination.png')),
            'E': PILImage.open(Path('images/gestures', 'hand_open.png')),
            'W': PILImage.open(Path('images/gestures', 'hand_close.png')),
            #'NE': cv2.VideoCapture(Path('images_master/videos', 'IMG_4930.MOV')),  # store path instead
            #'NW': cv2.VideoCapture(Path('images_master/videos', 'IMG_4931.MOV'))
        }
        # # ---- This is for when training with simultnoeus gestures gets implemented right ---
        # self.axis_media_paths = {
        #     'N': Path('images/gestures', 'pronation.png'),
        #     'S': Path('images/gestures', 'supination.png'),
        #     'E': Path('images/gestures', 'hand_open.png'),
        #     'W': Path('images/gestures', 'hand_close.png'),
        #     'NE': Path('images_master/videos', 'IMG_4930.MOV'), 
        #     'NW': Path('images_master/videos', 'IMG_4931.MOV')
        # }
        self.window = None
        self.initialize_ui()
        self.window.mainloop()

    def initialize_ui(self):
        '''
        Initializes the UI for the training protocol. This is the main menu for the training protocol.
        The order of execution is as follows:
        1. Choose which model you want. The suppored models are listed. Default is Regression with LR.
        2. Train the model. This will open a new window where you can choose training parameters, such as window size, window increment, repetition of training, etc. 
        3. After training you can configure and tune the model, such as adding deadband, gain and threshold angles.
        4. When model is trained and tuned, you can try the system with Isofitts or run the prosthesis. 
        '''
        # Create the simple menu UI:
        self.window = Tk()
        if not self.model_str:
            self.model_str = StringVar(value='LR')
        else:
            self.model_str = StringVar(value=self.model_str.get())
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.title("Game Menu")
        self.window.geometry("800x800") # Changed the size

        # Label 
        Label(self.window, font=("Arial bold", 20), text = 'Simultanous proportional prosthesis control').pack(pady=(10,20))
        # Train Model Button
        Button(self.window, font=("Arial", 18), text = 'Get Training Data', command=self.launch_training).pack(pady=(0,20))
        # Visualize predictions
        Button(self.window, font=("Arial", 18), text = 'Configure Model', command=self.configure_model_callback).pack(pady=(0,20))
        # Run prosthesis
        #Button(self.window, font=("Arial", 18), text = 'Run Prosthesis', command=self.run_prosthesis).pack(pady=(0,20)) # Added 22.04. This may be in Configure Model -> try it out.
        # Start Isofitts
        Button(self.window, font=("Arial", 18), text = 'Start Isofitts', command=self.start_test).pack(pady=(0,20))

        # Model Input
        self.model_type = IntVar()
        r1 = Radiobutton(self.window, text='Classification / Pattern Recognition', variable=self.model_type, value=1)
        r1.pack()
        r2 = Radiobutton(self.window, text='Regression / Proportional Control', variable=self.model_type, value=2)
        r2.pack()
        r2.select() # default to regression

        frame = Frame(self.window)
        Label(self.window, text="Model:", font=("Arial bold", 18)).pack(in_=frame, side=LEFT, padx=(0,10))
        Entry(self.window, font=("Arial", 18), textvariable=self.model_str).pack(in_=frame, side=LEFT)
        frame.pack(pady=(20,10))
       
        Label(self.window, text="Classifier Model Options: LDA, KNN, SVM, MLP, RF, QDA, NB", font=("Arial", 15)).pack(pady=(0,10))       
        Label(self.window, text="Regressor Model Options: LR, SVM, MLP, RF, GB", font=("Arial", 15)).pack(pady=(0,10))

    # ---------- Functions for the buttons in the menu -----------
    # Gotten from LibEMG Menu.py 
    def start_test(self):
        self.window.destroy()
        self.set_up_model()
        if self.regression_selected():
            controller = RegressorController()
            save_file = Path('results', self.model_str.get() + '_reg' + ".pkl").absolute().as_posix()
        else:
            controller = ClassifierController(output_format=self.model.output_format, num_classes=5)
            save_file = Path('results', self.model_str.get() + '_clf' + ".pkl").absolute().as_posix()
        
        config = FittsConfig(num_trials=16, save_file=save_file)
        ISOFitts(controller, config).run()
        # Its important to stop the model after the game has ended
        # Otherwise it will continuously run in a seperate process
        self.model.stop_running()
        self.initialize_ui()

    def launch_training(self):
        self.window.destroy()
        if self.regression_selected():
            args = {'media_folder': 'animation/', 'data_folder': Path('data', 'regression').absolute().as_posix(), 'rep_time': 50} # CHANGED 22.04. DON'T KNOW IF IT IS BEST TO HAVE SELF.ARGS OR ARGS
        else:
            args = {'media_folder': 'images/', 'data_folder': Path('data', 'classification').absolute().as_posix()}
        
        training_ui = GUI(self.odh, args=args, width=1000, height=1000, gesture_height=700, gesture_width=700)
        training_ui.download_gestures([1,2,3,4,5], "images/") # Downloading gestures from github repo. Videos for simultaneous gestures are located in images_master/videos
        self.create_animation()
        # maybe have a sepearte window where in GUI where you user can give parameters before making animation?           
        training_ui.start_gui()
        self.initialize_ui()

    def configure_model_callback(self):
        self.window.destroy()
        if self.regression_selected(): # Trenger vel i utgpkt ikke disse, da de ikke brukes i ML_GUI? -> TODO: burde hente de i ML_GUI 
            data_folder = Path('data', 'regression').absolute().as_posix()
        else:
            data_folder = Path('data', 'classification').absolute().as_posix()
        args = {'window_size':150, 'window_increment':50, 'deadband': 0.1, 'thr_angle_mf1': 45, 'thr_angle_mf2': 45, 'gain_mf1': 1, 'gain_mf2': 1}
        model_ui = ML_GUI(online_data_handler=self.odh, regression_selected=self.regression_selected(), model_str=self.model_str.get(), axis_media=self.axis_media, args=args, training_data_folder=data_folder)
        model_ui.start_gui()
        self.initialize_ui()
    
    # --------- Helper functions for the buttons in the menu -----------
    # Gotten from LibEMG Menu.py. Cannot choose model parameters in GUI. Could save the model configuration in this window and use it in this function?
    def set_up_model(self):
        # These parameters should be set and sotred in ConfigureModelGUI, in a json-file or something and read here. 
        WINDOW_SIZE = 150   
        WINDOW_INCREMENT = 50
        if self.regression_selected(): 
            data_folder = 'data/regression' 
        else: 
            data_folder = 'data/classification' # Added 22.04. Hardcoded, could find a way to store this?

        # Step 1: Parse offline training data CHANGED 22.04 -> CHECK THAT IT WORKS
        with open(data_folder + '/collection_details.json', 'r') as f:
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
        
        # TODO: add regex-filter for classification. Only support regression for now. Changed 22.04. The same regexfilter for both since data-folder is changed further up. 
        regex_filters = [
                RegexFilter(left_bound = data_folder + "/C_", right_bound="_R", values = [str(i) for i in range(num_motions)], description='classes'),
                RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = [str(i) for i in range(num_reps)], description='reps')
            ]
        if self.regression_selected():
            metadata_fetchers = [
                FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=motion_names, description='labels'), package_function=lambda meta, data: _match_metadata_to_data(meta, data, class_map) ) #package_function=lambda x, y: True)
            ]
            labels_key = 'labels'
            metadata_operations = {'labels': 'last_sample'}
        else:
            metadata_fetchers = None
            labels_key = 'classes'
            metadata_operations = None

        # Step 1: Parse offline training data THIS IS OLD, CHANGED IN MY CODE
        # if self.regression_selected():
        #     regex_filters = [
        #         RegexFilter(left_bound='regression/C_0_R_', right_bound='_emg.csv', values=['0', '1', '2'], description='reps')
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

        odh = OfflineDataHandler()
        odh.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",")
        train_windows, train_metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=metadata_operations)

        # Step 2: Extract features from offline data
        fe = FeatureExtractor()
        feature_list = fe.get_feature_groups()['HTD']
        training_features = fe.extract_features(feature_list, train_windows, array=True)

        # Step 3: Dataset creation
        data_set = {}
        data_set['training_features'] = training_features
        data_set['training_labels'] = train_metadata[labels_key]

        # Step 4: Create the EMG model
        model = self.model_str.get()
        print('Fitting model...')
        if self.regression_selected():
            emg_model = EMGRegressor(model=model)
            emg_model.fit(feature_dictionary=data_set)
            #emg_model.add_deadband(0.1) # Add a deadband to the regression model. Value below this threshold will be considered 0.
            self.model = OnlineEMGRegressor(emg_model, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list, std_out=True)
        else:
            emg_model = EMGClassifier(model=model)
            emg_model.fit(feature_dictionary=data_set)
            emg_model.add_velocity(train_windows, train_metadata[labels_key])
            self.model = OnlineEMGClassifier(emg_model, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list, output_format='probabilities', std_out=True)

        # Step 5: Create online EMG model and start predicting.
        print('Model fitted and running!')
        self.model.run(block=False) # block set to false so it will run in a seperate process.


    def create_animation(self):
        """Creates animations for different motor functions used in the training protocol."""
        for mf, (x_factor, y_factor) in self.motor_functions.items():
            output_filepath = Path(f'animation/collection_{mf}.mp4').absolute()
            if output_filepath.exists():
                print(f'Animation file for {mf} already exists. Skipping creation.')
                continue

            # Generate base movement
            base_motion = self._generate_base_signal()
            # Apply movement transformation
            x_coords = x_factor * base_motion
            y_coords = y_factor * base_motion

            # Calculate angles (in radians) for the arrow animator
            angles = np.arctan2(y_coords, x_coords)
            #angles = np.arctan2(y_factor, x_factor)
            # Apply movement transformation
            coordinates = np.hstack((x_coords, y_coords, angles))  
            
            #animator = MediaGridAnimator(output_filepath=output_filepath.as_posix(), show_direction=True, show_countdown=True, axis_media_paths=self.axis_media_paths, figsize=(10,10), normalize_distance=True, show_boundary=True, tpd=2)#, plot_line=True) # plot_line does not work
            #animator.plot_center_icon(coordinates, title=f'Regression Training - {mf}', save_coordinates=True, xlabel='MF 1', ylabel='MF 2')
            scatter_animator_x = ScatterPlotAnimator(output_filepath=output_filepath.as_posix(), show_direction=True, show_countdown=True, axis_images=self.axis_media, figsize=(10,10), normalize_distance=True, show_boundary=True, tpd=2)#, plot_line=True) # plot_line does not work
            scatter_animator_x.save_plot_video(coordinates, title=f'Regression Training - {mf}', save_coordinates=True, verbose=True)
            #arrow_animator = ArrowPlotAnimator(output_filepath=output_filepath.as_posix(), show_direction=True, show_countdown=True, axis_images=self.axis_media, figsize=(10,10), normalize_distance=True, show_boundary=True, tpd=2)#, plot_line=True) # plot_line does not work
            #arrow_animator.save_plot_video(coordinates, title=f'Regression Training - {mf}', save_coordinates=True, verbose=True)


    def _generate_base_signal(self):
        """Generates a sinusoidal motion profile for training animation."""
        period = 3      # period of sinusoid (seconds)
        cycles = 1      # number of reps of the motorfunction
        steady_time = 2 # how long you want to stay in end position (seconds)
        rest_time = 5   # time for rest between reps (seconds)
        fps = 24        # frames per second

        coordinates = []
        # Time vectors
        t_right = np.linspace(0, period / 2, int(fps * period / 2))#, endpoint=False)  # Rising part
        t_left = np.linspace(period / 2, period, int(fps * period / 2))#, endpoint=False)  # Falling part

        # Generate sine wave motion
        sin_right = np.sin(2 * np.pi * (1 / period) * t_right)  # 0 to 1
        sin_left = np.sin(2 * np.pi * (1 / period) * t_left)  # 1 to -1

        # Create steady phases
        steady_right = np.ones(fps * steady_time)  # Hold at max (1)
        steady_left = -np.ones(fps * steady_time)  # Hold at min (-1)

        # Build full movement cycle
        one_cycle = np.concatenate([
            sin_right[0:int(len(sin_right)/2)], steady_right, 
            sin_right[int(len(sin_right)/2):], sin_left[0:int(len(sin_right)/2)], 
            steady_left, sin_left[int(len(sin_right)/2):]])
        
        # Repeat for the number of cycles
        full_motion = np.tile(one_cycle, cycles)
        coordinates.append(full_motion)  # add sinusoids
        coordinates.append(np.zeros(fps * rest_time))   # add rest time

        # Convert into 2D (N x M) array with isolated sinusoids per DOF
        coordinates = np.expand_dims(np.concatenate(coordinates, axis=0), axis=1)
        return coordinates

    def regression_selected(self):  
        return self.model_type.get() == 2
    
    def on_closing(self):
        # Clean up all the processes that have been started
        self.window.destroy()

if __name__ == "__main__":
    train = TrainingProtocol()