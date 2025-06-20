

##### Made by Rikke 11.10.2024 #####


import shutil
from pathlib import Path
import dearpygui.dearpygui as dpg
import numpy as np
import os
from itertools import compress
import time
import csv
import json
from datetime import datetime
from ._utils import Media, set_texture, init_matplotlib_canvas, matplotlib_to_numpy

import threading
import matplotlib.pyplot as plt
from libemg.data_handler import OfflineDataHandler, RegexFilter
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics
from libemg.emg_predictor import EMGRegressor, OnlineEMGRegressor

class TrainAndCollectPanel:
    def __init__(self,
                 num_reps=3,
                 rise_time=2,
                 steady_state_time=1,
                 time_between_reps=1,
                 axis_media='media/',
                 data_folder='data/',
                 rest_between_sets=2,
                 auto_advance=False,
                 exclude_files=[],
                 gui = None,
                 video_player_width = 720,
                 video_player_height = 480):
        
        self.num_reps = num_reps
        self.rise_time = rise_time
        self.time_between_reps = time_between_reps
        self.steady_state_time = steady_state_time
        self.axis_media = axis_media
        self.data_folder = data_folder
        self.rest_between_sets = rest_between_sets
        self.auto_advance=auto_advance
        self.exclude_files = exclude_files
        self.gui = gui
        self.video_player_width = video_player_width
        self.video_player_height = video_player_height

        self.widget_tags = {"configuration":['__dc_configuration_window','__dc_num_reps','__dc_rest_between_sets', '__dc_media_folder',\
                                             '__dc_auto_advance', '__dc_rise_time', '__dc_steady_state_time', '__dc_time_between_reps'], #NOTE! Added signal times 10.05
                            "collection":   ['__dc_collection_window', '__dc_prompt_spacer', '__dc_prompt', '__dc_progress', '__dc_redo_button'],
                            "visualization": ['__vls_visualize_window']}
        

    def cleanup_window(self, window_name):
        widget_list = self.widget_tags[window_name]
        for w in widget_list:
            if dpg.does_alias_exist(w):
                dpg.delete_item(w)      
    
    def spawn_configuration_window(self):
        self.cleanup_window("configuration")
        self.cleanup_window("collection")
        self.cleanup_window("visualization")
        with dpg.window(tag="__dc_configuration_window", label="Data Collection Configuration"):
            
            dpg.add_text(label="Training Menu")
            with dpg.table(header_row=False, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):

                dpg.add_table_column(label="")
                dpg.add_table_column(label="")
                dpg.add_table_column(label="")
                # REP ROW
                with dpg.table_row(): 
                    with dpg.group(horizontal=True):
                        dpg.add_text("Num Reps: ")
                        dpg.add_input_text(default_value=self.num_reps,
                                        tag="__dc_num_reps",
                                        width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Time Per Rep")
                        dpg.add_input_text(default_value=self.rep_time,
                                        tag="__dc_rep_time",
                                        width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Time Between Reps")
                        dpg.add_input_text(default_value=self.rest_time,
                                        tag="__dc_rest_time", 
                                        width=100)
                # SIGNAL GENERATOR ROW - NOTE! Added signal times 10.05
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Rise Time:")
                        dpg.add_input_int(default_value=1, tag="__dc_rise_time", width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Steady-State Time:")
                        dpg.add_input_int(default_value=1, tag="__dc_steady_state_time", width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Rest Time:")
                        dpg.add_input_int(default_value=1, tag="__dc_signal_rest_time", width=100)
                # FOLDER ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Media Folder:")
                        dpg.add_input_text(default_value=self.media_folder, 
                                        tag="__dc_media_folder", width=250)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Output Folder:")
                        dpg.add_input_text(default_value=self.data_folder, 
                                        tag="__dc_output_folder",
                                        width=250)
                # CHECKBOX ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Auto-Advance")
                        dpg.add_checkbox(default_value=self.auto_advance,
                                        tag="__dc_auto_advance")
                # BUTTON ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Start Training", callback=self.start_callback)
                        dpg.add_button(label="Show EMG signal", callback=self.visualize_callback)
        
        # dpg.set_primary_window("__dc_configuration_window", True)

    def start_training_callback(self):
        if not (self.gui.online_data_handler and sum(list(self.gui.online_data_handler.get_data()[1].values()))):
            raise ConnectionError('Attempted to start data collection, but data are not being received. Please ensure the OnlineDataHandler is receiving data.')

        self.get_settings()
        dpg.delete_item("__dc_configuration_window")
        self.cleanup_window("configuration")
        self.set_media_list() # Made by me - 11.05

        self.spawn_collection_thread = threading.Thread(target=self.spawn_collection_window, args=(media_list,))
        self.spawn_collection_thread.start()
    
    def set_media_list(self): # Made by me - 11.05
        files = os.listdir(self.media_folder)
        label_files = [file for file in files if file.endswith(('.txt', '.csv'))]
        self.num_motions = len(label_files)
        # make the collection_details.json file
        collection_details = {}
        collection_details["num_motions"] = self.num_motions
        collection_details["num_reps"]    = self.num_reps
        collection_details["classes"] =   [f.split('.')[0] for f in label_files]
        collection_details["class_map"] = {index: f.split('.')[0] for index, f in enumerate(label_files)}
        collection_details["time"]    = datetime.now().isoformat()
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        with open(Path(self.output_folder, "collection_details.json").absolute().as_posix(), 'w') as f:
            json.dump(collection_details, f)

        collection_conf = []
        for rep_index in range(self.num_reps):
            for class_index, label_file in enumerate(label_files):
                label_str = Path(label_file).stem
                label_path = Path(self.media_folder, label_file).absolute().as_posix()
            
            # Load the label signal or function (you design this part)
            label_signal = self.load_label_signal(label_path)  # Could return a function or array

            # Estimate duration from length or define a fixed one
            duration = self.estimate_duration_from_label(label_path)

            collection_conf.append([label_signal, label_str, class_index, rep_index, duration])

        return collection_conf
    
    def start_callback(self):
        if not (self.gui.online_data_handler and sum(list(self.gui.online_data_handler.get_data()[1].values()))):
            raise ConnectionError('Attempted to start data collection, but data are not being received. Please ensure the OnlineDataHandler is receiving data.')

        self.get_settings()
        dpg.delete_item("__dc_configuration_window")
        self.cleanup_window("configuration")
        media_list = self.gather_media()

        self.spawn_collection_thread = threading.Thread(target=self.spawn_collection_window, args=(media_list,))
        self.spawn_collection_thread.start()
        # self.spawn_collection_window(media_list)

    def get_settings(self):
        self.num_reps      = int(dpg.get_value("__dc_num_reps"))
        self.rep_time      = float(dpg.get_value("__dc_rep_time"))
        self.rest_time     = float(dpg.get_value("__dc_rest_time"))
        self.rise_time     = float(dpg.get_value("__dc_rise_time"))             # Added by me 11.05
        self.steady_state_time = float(dpg.get_value("__dc_steady_state_time")) # Added by me 11.05
        self.signal_rest_time = float(dpg.get_value("__dc_signal_rest_time"))   # Added by me 11.05
        self.media_folder  = dpg.get_value("__dc_media_folder")
        self.output_folder = dpg.get_value("__dc_output_folder")
        self.auto_advance  = bool(dpg.get_value("__dc_auto_advance"))

    def gather_media(self):
        # find everything in the media folder
        files = os.listdir(self.media_folder)
        #files = sorted(files) # Consider removing this line, or sorting in another way
        labels_files = [file for file in files if file.endswith(('.txt', '.csv'))]
        files = [file for file in files if file.endswith((".gif",".png",".mp4","jpg"))]
        self.num_motions = len(files)
        collection_conf = []
        # make the collection_details.json file
        collection_details = {}
        collection_details["num_motions"] = self.num_motions
        collection_details["num_reps"]    = self.num_reps
        collection_details["classes"] =   [f.split('.')[0] for f in files]
        collection_details["class_map"] = {index: f.split('.')[0] for index, f in enumerate(files)}
        collection_details["time"]    = datetime.now().isoformat()
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        with open(Path(self.output_folder, "collection_details.json").absolute().as_posix(), 'w') as f:
            json.dump(collection_details, f)

        for media_file in files:
            matching_labels_files = [labels_file for labels_file in labels_files if Path(labels_file).stem == Path(media_file).stem]
            if len(matching_labels_files) == 1:
                # Copy labels file to data directory
                labels_file = matching_labels_files[0]
                class_index = [idx for idx, filename in collection_details['class_map'].items() if filename == Path(labels_file).stem]
                assert len(class_index) == 1, f"Expected a single matching filename in collection_details.json, but got {len(class_index)} for {labels_file}."
                class_index = class_index[0]
                labels_new_filename = Path(labels_file).with_stem(f"C_{class_index}").name
                shutil.copy(Path(self.media_folder, labels_file).absolute(), Path(self.data_folder, labels_new_filename).absolute())

        # make the media list for SGT progression
        for rep_index in range(self.num_reps):
            for class_index, motion_class in enumerate(files):
                # entry for collection of rep
                media = Media()
                print(f"Loading {motion_class} in: ", Path(self.media_folder, motion_class).absolute().as_posix())
                media.from_file(Path(self.media_folder, motion_class).absolute().as_posix())

                if media.type in ('mp4', 'gif'):
                    # Automatically calculate length of video
                    rep_time = media.n_frames / media.fps
                else:
                    rep_time = self.rep_time
                collection_conf.append([media, motion_class.split('.')[0], class_index, rep_index, rep_time])
        return collection_conf

    def spawn_collection_window(self, media_list):
        # open first frame of gif
        self.gui.online_data_handler.prepare_smm()
        #texture = media_list[0][0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height)
        #set_texture("__dc_collection_visual", texture, width=self.video_player_width, height=self.video_player_height)
        
        collection_window_width  = self.video_player_width + 100
        collection_window_height = self.video_player_height + 300
        with dpg.window(label="Collection Window",
                        tag="__dc_collection_window",
                        width=collection_window_width,
                        height=collection_window_height):
            
            with dpg.group(horizontal=True):
                dpg.add_spacer(height=20,width=self.video_player_width/2+30-(7*len("Collection Menu"))/2)
                dpg.add_text(default_value="Collection Menu")
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_rep_spacer",height=20,width=self.video_player_width/2+30 - (7*len(media_list[0][1]))/2)
                dpg.add_text(f"Rep 1 of {self.num_reps}", tag="__dc_rep")
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_prompt_spacer",height=20,width=self.video_player_width/2+30 - (7*len(media_list[0][1]))/2)
                dpg.add_text(media_list[0][1], tag="__dc_prompt")
            with dpg.group(horizontal=True):
                dpg.add_spacer(height=20,width=30)
                dpg.add_image("__dc_collection_visual")
            with dpg.group(horizontal=True):
                dpg.add_spacer(height=20,width=30)
                dpg.add_progress_bar(tag="__dc_progress", default_value=0.0,width=self.video_player_width)
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_redo_spacer", height=20, width=self.video_player_width/2+30 - (7*len("Redo"))/2)
                dpg.add_button(tag="__dc_redo_button", label="Redo", callback=self.redo_collection_callback)
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_continue_spacer", height=20, width=self.video_player_width/2+30 - (7*len("Continue"))/2)
                dpg.add_button(tag="__dc_continue_button", label="Continue", callback=self.continue_collection_callback)
            with dpg.group(horizontal=True): # Added by me - meant for showing plot of data collected after one rep
                dpg.add_spacer(tag="__dc_recent_spacer", height=20, width=self.video_player_width/2+30 - (7*len("Show Recent Data"))/2)
                dpg.add_button(tag="__dc_recent_data", label="Show Recent Data", callback=self.recent_data_callback)
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_features_spacer", height=20, width=self.video_player_width/2+30 - (7*len("Visualize Features"))/2)
                dpg.add_button(tag="__dc_features", label="Visualize Features", callback=self.visualize_features_callback)
            dpg.hide_item(item="__dc_redo_button")
            dpg.hide_item(item="__dc_continue_button")
            dpg.hide_item(item="__dc_recent_data") # Added by me
            dpg.hide_item(item="__dc_features")
            
        
        # dpg.set_primary_window("__dc_collection_window", True)

        self.run_sgt(media_list)
        # clean up the window
        
        dpg.delete_item("__dc_collection_window")
        self.cleanup_window("collection")
        # open config back up
        self.spawn_configuration_window()

    def run_sgt_real_time(self, axis_media):
        '''
        Make training data collection in real time, i.e. a plot depending on given inputs in GUI.
        Parameters:
            axis_media: Dictionary with keys as the direction and values as the media objects.'''
        self.i = 0
        self.advance = True
        self.gui.online_data_handler.reset()
        # Fetch GUI settings
        rise_time = float(dpg.get_value("__dc_rise_time"))
        hold_time = float(dpg.get_value("__dc_steady_state_time"))
        signal_rest_time = float(dpg.get_value("__dc_signal_rest_time"))

        while self.i < len(axis_media):
            self.rep_buffer = {mod:[] for mod in self.gui.online_data_handler.modalities}
            self.rep_count  = {mod:0 for mod in self.gui.online_data_handler.modalities}
            # do the rest
            if self.rest_time and self.i < len(axis_media):
                self.play_collection_visual(media_list[self.i], active=False)
                media_list[self.i][0].reset()
            
            self.gui.online_data_handler.reset()
            
            self.play_motion_plot()
            output_path = Path(self.output_folder, "C_" + str(media_list[self.i][2]) + "_R_" + str(media_list[self.i][3]) + ".csv").absolute().as_posix()

            self.play_collection_visual(media_list[self.i], active=True)
            
            output_path = Path(self.output_folder, "C_" + str(media_list[self.i][2]) + "_R_" + str(media_list[self.i][3]) + ".csv").absolute().as_posix()
            self.save_data(output_path)
            last_rep = media_list[self.i][3]
            self.i = self.i+1
            is_final_media = self.i == len(media_list)
            if is_final_media:
                # At the end of the list, so we must be finished a rep
                rep_is_finished = True
                current_rep = self.num_reps - 1
            else:
                # Check if we've finished a rep
                current_rep = media_list[self.i][3]
                rep_is_finished = last_rep != current_rep

            # pause / redo goes here!
            if rep_is_finished  or (not self.auto_advance):
                # Show redo / continue buttons
                dpg.show_item(item="__dc_redo_button")
                dpg.show_item(item="__dc_continue_button")
                dpg.set_item_user_data("__dc_recent_data", (media_list[self.i-1][2], media_list[self.i-1][3], media_list[self.i-1][4]))

                
    def run_sgt(self, media_list):
        self.i = 0
        self.advance = True
        self.gui.online_data_handler.reset()
        while self.i < len(media_list):
            self.rep_buffer = {mod:[] for mod in self.gui.online_data_handler.modalities}
            self.rep_count  = {mod:0 for mod in self.gui.online_data_handler.modalities}
            # do the rest
            if self.rest_time and self.i < len(media_list):
                self.play_collection_visual(media_list[self.i], active=False)
                media_list[self.i][0].reset()
            self.gui.online_data_handler.reset()
            
            self.play_collection_visual(media_list[self.i], active=True)
            
            output_path = Path(self.output_folder, "C_" + str(media_list[self.i][2]) + "_R_" + str(media_list[self.i][3]) + ".csv").absolute().as_posix()
            self.save_data(output_path)
            last_rep = media_list[self.i][3]
            self.i = self.i+1
            is_final_media = self.i == len(media_list)
            if is_final_media:
                # At the end of the list, so we must be finished a rep
                rep_is_finished = True
                current_rep = self.num_reps - 1
            else:
                # Check if we've finished a rep
                current_rep = media_list[self.i][3]
                rep_is_finished = last_rep != current_rep

            # pause / redo goes here!
            if rep_is_finished  or (not self.auto_advance):
                # Show redo / continue buttons
                self.advance = False
                dpg.show_item(item="__dc_redo_button")
                dpg.show_item(item="__dc_continue_button")
                dpg.show_item(item="__dc_recent_data") # Added by me
                dpg.set_item_user_data("__dc_recent_data", (media_list[self.i-1][2], media_list[self.i-1][3], media_list[self.i-1][4])) # Added by me . Give the button the class index, rep index, and rep time for the visualize function
                dpg.show_item(item="__dc_features") # Added by me - meant for showing plot of data collected after one rep
                
                while not self.advance:
                    time.sleep(0.1)
                    dpg.configure_app(manual_callback_management=True)
                    jobs = dpg.get_callback_queue()
                    dpg.run_callbacks(jobs)
                dpg.configure_app(manual_callback_management=False)
                if not is_final_media:
                    dpg.set_value('__dc_rep', value=f"Rep {media_list[self.i][3] + 1} of {self.num_reps}")
        
    def redo_collection_callback(self):
        if self.auto_advance:
            self.i      = self.i - self.num_motions
        else:
            self.i      = self.i - 1 
        dpg.hide_item(item="__dc_redo_button")
        dpg.hide_item(item="__dc_continue_button")
        dpg.hide_item(item="__dc_recent_data") # Added by me 
        dpg.hide_item(item="__dc_features") # Added by me - meant for showing plot of data collected after one rep
        self.advance = True
    
    def continue_collection_callback(self):
        dpg.hide_item(item="__dc_redo_button")
        dpg.hide_item(item="__dc_continue_button")
        dpg.hide_item(item="__dc_recent_data") # Added by me 
        dpg.hide_item(item="__dc_features") # Added by me - meant for showing plot of data collected after one rep
        self.advance = True

    # Added by me
    def recent_data_callback(self): # Added by me - meant for showing plot of data collected after one rep
        dataset_folder = self.output_folder
        class_num, rep_num, sampling_time = dpg.get_item_user_data("__dc_recent_data")
        if class_num is None or rep_num is None:
            raise ValueError("No data to visualize")
        regex_filters = [ 
            RegexFilter(left_bound=f'/C_{class_num}_R_', right_bound='_emg.csv', values=[str(rep_num)], description='reps')
        ]

        self.offline_dh = OfflineDataHandler()
        self.offline_dh.get_data(folder_location=dataset_folder, regex_filters=regex_filters, delimiter=",")

        self._plot_thread = threading.Thread(target=self._plot_data_helper, args=(f"Recent Data - Class: {class_num}, Rep: {rep_num}", sampling_time, ))
        self._plot_thread.start()

    # Added by me 
    def _plot_data_helper(self, title, time):
        self.offline_dh.visualize(block=False, title=title, time=time)

    def visualize_features_callback(self):
        self.WINDOW_SIZE, self.WINDOW_INCREMENT = 150, 100 # TODO: Change these later
        dataset_folder = self.output_folder
        class_num, rep_num, sampling_time = dpg.get_item_user_data("__dc_recent_data")
        if class_num is None or rep_num is None:
            raise ValueError("No data to visualize")
        regex_filters = [ 
            RegexFilter(left_bound=f'/C_{class_num}_R_', right_bound='_emg.csv', values=[str(rep_num)], description='reps')
        ]
        #metadata_operations = {'labels': 'last_sample'}
        self.offline_dh = OfflineDataHandler()
        self.offline_dh.get_data(folder_location=dataset_folder, regex_filters=regex_filters, delimiter=",")
        train_windows, train_metadata = self.offline_dh.parse_windows(self.WINDOW_SIZE, self.WINDOW_INCREMENT, metadata_operations=None)

        fe = FeatureExtractor()
        print("Extracting features")
        #feature_list = fe.get_feature_list() # Get all available features - TODO: Could be made a GUI option
        feature_list = fe.get_feature_groups()['HTD'] # Make this chosen from the GUI later
        feature_dict = {}
        for feature in (feature_list):
            train_features = fe.extract_features([feature], train_windows, array=True)
            feature_dict[feature] = train_features
            #feature_dict = {"training_features": train_features,
            #                "training_labels": train_metadata['labels']}

        self.feature_plot_thread = threading.Thread(target=self._plot_features_helper, args=(fe, feature_dict, sampling_time, ))
        self.feature_plot_thread.start()
        self.feature_plot_thread.join()

    def _plot_features_helper(self, fe, fe_dict, sampling_time):
        fe.visualize_with_time(fe_dict, sampling_time=sampling_time, block=False)

    def play_collection_visual(self, media, active=True):
        if active:
            timer_duration = media[-1]
            dpg.set_value("__dc_prompt", value=media[1])
            dpg.set_item_width("__dc_prompt_spacer",width=self.video_player_width/2+30 - (7*len(media[1]))/2)
        else:
            timer_duration = self.rest_time
            dpg.set_value("__dc_prompt", value="Up next: "+media[1])
            dpg.set_item_width("__dc_prompt_spacer",width=self.video_player_width/2+30 - (7*len("Up next: "+media[1]))/2)
        
        
        texture = media[0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height, grayscale=not(active))
        set_texture("__dc_collection_visual", texture, self.video_player_width, self.video_player_height)
        # initialize motion and frame timers
        motion_timer = time.perf_counter_ns()
        while (time.perf_counter_ns() - motion_timer)/1e9 < timer_duration:
            time.sleep(1/media[0].fps) # never refresh faster than media fps
            # update visual
            media[0].advance_to((time.perf_counter_ns() - motion_timer)/1e9)
            texture = media[0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height, grayscale=not(active))
            set_texture("__dc_collection_visual", texture, self.video_player_width, self.video_player_height)
            # update progress bar
            progress = min(1,(time.perf_counter_ns() - motion_timer)/(1e9*timer_duration))
            # grab incoming new data
            if active:
                vals, count = self.gui.online_data_handler.get_data()
                for mod in self.gui.online_data_handler.modalities:
                    new_samples = count[mod][0][0]-self.rep_count[mod]
                    self.rep_buffer[mod] = [vals[mod][:new_samples,:]] + self.rep_buffer[mod]
                    self.rep_count[mod]  = self.rep_count[mod] + new_samples

            dpg.set_value("__dc_progress", value = progress)        
    
    def save_data(self, filename):
        file_parts = filename.split('.')
        
        for mod in self.rep_buffer:
            filename = file_parts[0] + "_" + mod + "." + file_parts[1]
            data = np.vstack(self.rep_buffer[mod])[::-1,:]
            if data.size == 0:
                raise ConnectionError('Attempting to store data, but received 0 samples during repetition, suggesting that the data stream from the device has been interrupted. Please check the device connection and verify that previous files are not missing samples.')
            with open(filename, "w", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for row in data:
                    writer.writerow(row)

    def visualize_callback(self):
        self.visualization_thread = threading.Thread(target=self._run_visualization_helper)
        self.visualization_thread.start()
    
    def _run_visualization_helper(self):
        self.gui.online_data_handler.visualize(block=False)
