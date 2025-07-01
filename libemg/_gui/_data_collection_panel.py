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
from libemg.filtering import Filter
from libemg.animator import ScatterPlotAnimator

class DataCollectionPanel:
    def __init__(self,
                 num_reps=3,
                 rep_time=3,
                 media_folder='media/',
                 data_folder='data/',
                 rest_time=2,
                 auto_advance=True,
                 exclude_files=[],
                 gui = None,
                 video_player_width = 720,
                 video_player_height = 480):
        
        self.num_reps = num_reps
        self.rep_time = rep_time
        self.media_folder = media_folder
        self.data_folder  = data_folder # Changed this 12.05 (Rikke), is called data_folder in GUI but data_folder in DataCollectionPanel
        self.rest_time = rest_time
        self.auto_advance=auto_advance
        self.exclude_files = exclude_files
        self.gui = gui
        self.video_player_width = video_player_width
        self.video_player_height = video_player_height

        self.widget_tags = {"configuration":['__dc_configuration_window','__dc_num_reps','__dc_rep_time','__dc_rest_time', '__dc_media_folder',\
                                             '__dc_auto_advance','__dc_rise_time', '__dc_steady_state_time'], #NOTE! Added signal times 10.05 
                            "collection":   ['__dc_collection_window', '__dc_prompt_spacer', '__dc_prompt', '__dc_progress', '__dc_redo_button'],
                            "visualization": ['__vls_visualize_window', '__dc_recent_data']}
        self.overwrite_flags = {}  # NEW: Track user choice for each motion class
        self.pending_motion_classes = []  # NEW: Queue of motion classes to process

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

                dpg.add_table_column(label="", init_width_or_weight=2.0)
                dpg.add_table_column(label="", init_width_or_weight=1.0)
                dpg.add_table_column(label="", init_width_or_weight=1.0)

                # Check current mode at rendering time. This is to give different inputs depending on whether regression or classification is selected.
                if self.gui and self.gui.regression_selected:
                    rep_label = "Time to Max Contraction"
                    rest_label = "Hold Duration at Max"
                    rep_tag = "__dc_time_to_max"
                    rest_tag = "__dc_hold_duration"
                else:   
                    rep_label = "Time Per Rep"
                    rest_label = "Time Between Reps"
                    rep_tag = "__dc_rep_time"
                    rest_tag = "__dc_rest_time"
                # REP ROW
                with dpg.table_row(): 
                    with dpg.group(horizontal=True):
                        dpg.add_text("Num Reps: ")
                        dpg.add_input_text(default_value=self.num_reps,
                                        tag="__dc_num_reps",
                                        width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text(rep_label)
                        dpg.add_input_text(default_value=self.rep_time,
                                        tag=rep_tag,
                                        width=100)
                    with dpg.group(horizontal=True):
                        dpg.add_text(rest_label)
                        dpg.add_input_text(default_value=self.rest_time,
                                        tag=rest_tag,
                                        width=100)
                # FOLDER ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Media Folder:")
                        dpg.add_input_text(default_value=self.media_folder, 
                                        tag="__dc_media_folder", width=250)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Output Folder:")
                        dpg.add_input_text(default_value=self.data_folder, 
                                        tag="__dc_data_folder",
                                        width=350)
                # CHECKBOX ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Auto-Advance")
                        dpg.add_checkbox(default_value=self.auto_advance,
                                        tag="__dc_auto_advance")
                # BUTTON ROW
                with dpg.table_row():
                    with dpg.group(horizontal=True):
                        if self.gui.regression_selected:
                            dpg.add_button(label="Create Training Animation", callback=self.create_animation_callback)
                        dpg.add_button(label="Start Training", callback=self.start_callback)
                        dpg.add_button(label="Show EMG signal", callback=self.visualize_callback)
        
        # dpg.set_primary_window("__dc_configuration_window", True)

    def create_animation_callback(self):
        self.get_settings()
        self.pending_motion_classes = list(self.gui.motion_classes.items())  # Always fresh start
        self.process_next_motion_class()
         
    
    def process_next_motion_class(self):
        if not hasattr(self, "pending_motion_classes") or self.pending_motion_classes is None:
            self.pending_motion_classes = list(self.gui.motion_classes.items())

        if not self.pending_motion_classes:
            print("All animations processed.")
            self.pending_motion_classes = None  # Reset for next run
            return

        fps = 24
        media_folder = Path(dpg.get_value("__dc_media_folder")).absolute()
                
        if not media_folder or not media_folder.exists():
            raise FileNotFoundError(f"Media folder {media_folder} does not exist. Please specify a valid media folder.")
         
        print("Processing next motion class...")
        motion_class, (x_factor, y_factor) = self.pending_motion_classes[0]
        print(f"Processing motion class: {motion_class}")

        output_filepath = media_folder / f'{motion_class}.mp4'

        if output_filepath.exists():
            print(f"Animation file for {motion_class} already exists at {output_filepath.as_posix()}.")
           # popup_tag = "__dc_animation_exists_popup"
            popup_tag = f"__dc_animation_exists_popup_{motion_class}"
            if dpg.does_item_exist(popup_tag):
                dpg.delete_item(popup_tag)

            print(f"Showing popup for existing animation file: {output_filepath.as_posix()}")
            with dpg.window(label="Animation Exists", modal=True, tag=popup_tag, no_resize=True, no_move=True):
                dpg.add_text(f"Animation file for motion class '{motion_class}' already exists.", tag=f"{popup_tag}_text1")
                dpg.add_text("Would you like to use the existing file or create a new one?", tag=f"{popup_tag}_text2")
                dpg.add_button(label="Use Existing", callback=lambda: self.handle_popup_choice("use_existing", output_filepath=None, popup_tag=popup_tag), tag=f"{popup_tag}_btn1")
                dpg.add_button(label="Create New", callback=lambda: self.handle_popup_choice("create_new", output_filepath=output_filepath, popup_tag=popup_tag), tag=f"{popup_tag}_btn2")
            return
        
        # If no file exists, create it
        self._create_animation_for_motion_class(motion_class, x_factor, y_factor, output_filepath, fps)


    def handle_popup_choice(self, choice, output_filepath, popup_tag):
        motion_class, (x_factor, y_factor) = self.pending_motion_classes[0]
        fps = 24  # Default FPS for animation

        # Close the popup first regardless of choice
        if dpg.does_item_exist(popup_tag):
            dpg.delete_item(popup_tag)
            print(f"Popup {popup_tag} deleted")

        if choice == "create_new":
            print(f"Overwriting and creating new animation for {motion_class} at {output_filepath.as_posix()}")
            self._create_animation_for_motion_class(motion_class, x_factor, y_factor, output_filepath, fps)

        elif choice == "use_existing":
            print(f"Using existing animation for {motion_class}.")
            # Remove the processed motion class from the queue and continue
            self.pending_motion_classes.pop(0)
        
            print(f"Removed {motion_class} from queue. Remaining motion classes: {len(self.pending_motion_classes)}")
        
            # Use frame callback to continue processing after popup is fully closed
            #dpg.set_frame_callback(3, lambda: self.process_next_motion_class())
            # Continue processing the next motion class
            self.process_next_motion_class()
        
        #if dpg.does_item_exist("__dc_animation_exists_popup"):
        #    dpg.delete_item("__dc_animation_exists_popup")
    


    def _create_animation_for_motion_class(self, motion_class, x_factor, y_factor, output_filepath, fps):
        """Helper method to create animation for a motion class"""
        print(f"Creating animation for {motion_class} at {output_filepath.as_posix()}")
        base_motion = self._generate_sawtooth_signal(rise_time=self.rep_time, 
                                                    hold_time=self.rest_time, 
                                                    rest_time=0, 
                                                    n_repeats=1, 
                                                    sampling_rate=fps, 
                                                    amplitude=1
                                                    ) # 1 second rise time, 3 seconds rest time, 1 repeat, 24 fps
        # Apply movement transformation
        x_coords = x_factor * base_motion
        y_coords = y_factor * base_motion

        coordinates = np.stack((x_coords, y_coords), axis=1)

        scatter_animator_x = ScatterPlotAnimator(output_filepath=output_filepath.as_posix(),
                                                show_direction=True, 
                                                show_countdown=True, 
                                                axis_images=self.gui.axis_media, 
                                                figsize=(10,10), 
                                                normalize_distance=False, 
                                                show_boundary=True, 
                                                fps=fps
                                                )# ,(tpd=5 this does not make any diffrence..)#, plot_line=True) # plot_line does not work
        scatter_animator_x.save_plot_video(coordinates, title=f'Regression Training - {motion_class}', save_coordinates=True, verbose=True)

        # Move to next motion class after creation
        self.pending_motion_classes.pop(0)
        self.process_next_motion_class()  # Continue to next motion class


    def start_callback(self):
        if not (self.gui.online_data_handler and sum(list(self.gui.online_data_handler.get_data()[1].values()))):
            raise ConnectionError('Attempted to start data collection, but data are not being received. Please ensure the OnlineDataHandler is receiving data.')

        self.get_settings()
        dpg.delete_item("__dc_configuration_window")
        self.cleanup_window("configuration")
        media_list = self.gather_media()

        self.spawn_collection_thread = threading.Thread(target=self.spawn_collection_window, args=(media_list,))
        self.spawn_collection_thread.start()

    def get_settings(self):
        self.num_reps      = int(dpg.get_value("__dc_num_reps"))
        self.media_folder  = dpg.get_value("__dc_media_folder")
        self.data_folder = dpg.get_value("__dc_data_folder")
        self.auto_advance  = bool(dpg.get_value("__dc_auto_advance"))
        if self.gui and self.gui.regression_selected:
            # If regression is selected, we use different tags for rep time and rest time
            self.rep_time = float(dpg.get_value("__dc_time_to_max"))
            self.rest_time = float(dpg.get_value("__dc_hold_duration"))
        else:
            self.rep_time = float(dpg.get_value("__dc_rep_time"))
            self.rest_time = float(dpg.get_value("__dc_rest_time"))

    def _generate_sawtooth_signal(self, rise_time, hold_time, rest_time, n_repeats, sampling_rate, amplitude=1):
        """
        Generate a sawtooth signal that rises linearly over 'rise_time' seconds, then rests flat for 'rest_time' seconds.

        Parameters
        ----------
        rise_time : float
            Duration of the rising edge (in seconds).
        rest_time : float
            Duration of the rest period after each rise (in seconds).
        n_repeats : int
            Number of sawtooth cycles to generate.
        sampling_rate : int
            Samples per second (Hz).
        amplitude : float
            Peak value of the signal.

        Returns
        -------
        signal : np.ndarray
            The generated sawtooth signal.
        time_vec : np.ndarray
            Corresponding time vector in seconds.
        """
        # Number of samples for rise and rest
        rise_samples = int(rise_time * sampling_rate)
        hold_samples = int(hold_time * sampling_rate)
        rest_samples = int(rest_time * sampling_rate)

        # Create one cycle: rise + rest
        rise_part = np.linspace(0, amplitude, rise_samples, endpoint=False)
        hold_part = np.full(hold_samples, amplitude)
        rest_part = np.zeros(rest_samples)

        # One full cycle
        cycle = np.concatenate([rise_part, hold_part, rest_part])

        # Repeat the cycle
        signal = np.tile(cycle, n_repeats)

        return signal
    
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
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        with open(Path(self.data_folder, "collection_details.json").absolute().as_posix(), 'w') as f:
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
        texture = media_list[0][0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height)
        set_texture("__dc_collection_visual", texture, width=self.video_player_width, height=self.video_player_height)
        
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
                dpg.add_button(tag="__dc_features", label="Visualize Features", callback=self.open_feature_selection_popup)
            dpg.hide_item(item="__dc_redo_button")
            dpg.hide_item(item="__dc_continue_button")
            dpg.hide_item(item="__dc_recent_data") # Added by me
            dpg.hide_item(item="__dc_features")
            
        
        # dpg.set_primary_window("__dc_collection_window", True)

        #self.run_sgt(media_list)
        self.run_sgt_regression(media_list) # By Rikke. Updated for optimizing the SGT when using regression.
    
        # clean up the window
        dpg.delete_item("__dc_collection_window")
        self.cleanup_window("collection")
        # open config back up
        self.spawn_configuration_window()

    def run_sgt_regression(self, media_list):
        self.gui.online_data_handler.reset()
        self.advance = True
        self.redo_class = None  # Reset redo flag
        # TODO: Hard coded "rest" - need to be changed
        rest_media = [m for m in media_list if m[1] == "rest"]
        if rest_media: rest_media = rest_media[0] # Only need one rest media
        sorted_classes = sorted(set(m[2] for m in media_list)) # Train each class with every reps before moving to next.
        for current_class in sorted_classes:
            self.last_class = current_class
            while True:         
                for rep_index in range(self.num_reps):
                    # Get correct media tuple for this class and rep
                    current_media = [m for m in media_list if m[2] == current_class and m[3] == rep_index][0]
                    dpg.set_value('__dc_rep', value=f"Rep {rep_index + 1} of {self.num_reps}")

                    if self.rest_time and not(current_media[1] == "rest" and self.auto_advance): # TODO: Hard coded "rest" - need to be changed
                        # Play rest media
                        self.play_collection_visual(rest_media, active=False, next_motion=current_media[1])
                        current_media[0].reset()
                        self.gui.online_data_handler.reset()
                    
                    # Reset rep buffer and count
                    self.rep_buffer = {mod: [] for mod in self.gui.online_data_handler.modalities}
                    self.rep_count = {mod: 0 for mod in self.gui.online_data_handler.modalities}

                    self.play_collection_visual(current_media, active=True)

                    # Save data
                    output_path = Path(self.data_folder, f"C_{current_class}_R_{rep_index}.csv").absolute().as_posix()
                    self.save_data(output_path)

                    if not self.auto_advance or rep_index == self.num_reps - 1: # Show redo / continue buttons if auto advance is off or if this is the last rep of the class
                        self.advance = False
                        dpg.show_item("__dc_redo_button")
                        dpg.show_item("__dc_continue_button")
                        dpg.show_item("__dc_recent_data")
                        dpg.set_item_user_data("__dc_recent_data", (current_media[1], current_media[2], rep_index, current_media[4]))
                        dpg.show_item("__dc_features")
                        if rep_index == self.num_reps - 1:
                            next_class = sorted_classes[current_class + 1] if current_class + 1 < len(sorted_classes) else self.num_motions
                            next_motion = [media[1] for media in media_list if media[2] == next_class][0] if current_class + 1 < len(sorted_classes) else "End of training"
                            dpg.set_value("__dc_rep", value="Up next: "+ next_motion)

                    while not self.advance:
                        time.sleep(0.1)
                        dpg.configure_app(manual_callback_management=True)
                        jobs = dpg.get_callback_queue()
                        dpg.run_callbacks(jobs)
                    dpg.configure_app(manual_callback_management=False)
                
                if self.redo_class == current_class:
                    self.redo_class = None  # Reset and repeat
                    continue
                break

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
            
            output_path = Path(self.data_folder, "C_" + str(media_list[self.i][2]) + "_R_" + str(media_list[self.i][3]) + ".csv").absolute().as_posix()
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
                dpg.set_item_user_data("__dc_recent_data", (media_list[self.i-1][1], media_list[self.i-1][2], media_list[self.i-1][3], media_list[self.i-1][4])) # Added by me . Give the button the edit 18.05: also give class label, class index, rep index, and rep time for the visualize function
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
        # Redo logic
        if hasattr(self, 'i'):  # This means we're in run_sgt (loop through all classes in one repetition (self.i) before moving to next repetition)
            if self.auto_advance:
                self.i = self.i - self.num_motions
            else:
                self.i = self.i - 1
        else:  # We're in run_sgt_regression (run through all repetitions of one class before moving to next class (self.last_class))
            if hasattr(self, 'last_class') and self.last_class is not None:
                # Set a flag to re-run this class
                self.redo_class = self.last_class
            else:
                print("Warning: last_class not set. Cannot redo.")
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
        dataset_folder = self.data_folder
        class_label, class_num, rep_num, sampling_duration = dpg.get_item_user_data("__dc_recent_data")
        if class_num is None:
            raise ValueError("No class specified for recent data visualization.")

        self.offline_dh = OfflineDataHandler()
        class_name_to_num = {class_label: class_num}
        # Case 1: run_sgt_regression with auto_advance → show all reps for this class
        if not hasattr(self, 'i') and self.auto_advance: # and not self.num_reps == 1:
            # All reps from this class
            regex_filters = [
                RegexFilter(
                    left_bound=f'/C_{class_num}_R_', 
                    right_bound='_emg.csv', 
                    values=[str(r) for r in range(self.num_reps)],
                    description='reps'
                )
            ]
            title = f"Recent Data - Class: {class_label}, All Reps"
        
        # Case 2: run_sgt OR manual step-through → show single rep
        else:
            if rep_num is None:
                raise ValueError("No repetition number provided for recent data visualization.")
            regex_filters = [
                RegexFilter(
                    left_bound=f'/C_{class_num}_R_', 
                    right_bound='_emg.csv', 
                    values=[str(rep_num)],
                    description='single rep'
                )
            ]
            title = f"Recent Data - Class: {class_num}, Rep: {rep_num+1}"

        self.offline_dh.get_data(folder_location=dataset_folder, regex_filters=regex_filters, delimiter=",")
        fi = Filter(2000)
        fi.install_filters({ "name": "highpass", "cutoff": 20, "order":2})
        fi.filter(self.offline_dh) # Filter to remove DC offset
        self._plot_thread = threading.Thread(target=self._plot_data_helper, args=(title, sampling_duration))
        self._plot_thread.start()

    def _plot_data_helper(self, title, duration):
        self.offline_dh.visualize_for_training(block=False, title=title, recording_duration=duration)

    def open_feature_selection_popup(self):
        all_features = FeatureExtractor().get_feature_list()
        with dpg.window(label="Select Features", tag="__dc_feature_selection_popup", popup=True, modal=True, no_title_bar=False, width=400, height=500) as popup_id:
            # Store checkbox values
            self.selected_features = {}
            with dpg.child_window(height=200, width=380, autosize_x=False, autosize_y=False):
                for feature in all_features:
                    self.selected_features[feature] = dpg.add_checkbox(label=feature, default_value=False, tag=f"feature_{feature}")
            
            # Input fields for window size and increment
            dpg.add_separator()
            dpg.add_text("Data Segmentation Settings:")
            self.window_size_input = dpg.add_input_int(label="Window Size", default_value=200, width=150)
            self.window_increment_input = dpg.add_input_int(label="Window Increment", default_value=100, width=150)

            dpg.add_spacer(height=10)
            # OK and Cancel buttons
            with dpg.group(horizontal=True):
                dpg.add_button(label="OK", callback=self._on_features_ok)
                dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("__dc_feature_selection_popup"))

    def _on_features_ok(self):
        # Get selected features
        chosen_features = [
            feature for feature, checkbox in self.selected_features.items()
            if dpg.get_value(checkbox)
        ]

        # Get window size and increment
        window_size = dpg.get_value(self.window_size_input)
        window_increment = dpg.get_value(self.window_increment_input)
        dpg.delete_item("__dc_feature_selection_popup")  # Close the popup
        # Call visualize with user-selected parameters
        self.visualize_features_callback(chosen_features, window_size, window_increment)

    def visualize_features_callback(self, chosen_features, window_size, window_increment):
        self.WINDOW_SIZE, self.WINDOW_INCREMENT = window_size, window_increment
        dataset_folder = self.data_folder
        class_label, class_num, rep_num, sampling_time = dpg.get_item_user_data("__dc_recent_data")

        ## NOTE! The same result happens if we extract data from all repetions or only one rep.
        if class_num is None:
            raise ValueError("No data to visualize")
        class_name_to_num = {class_label: class_num}

        if not hasattr(self, 'i') and self.auto_advance: # To ensure we are in run_sgt_regression and not run_sgt -> these works differently
            # All reps from this class
            regex_filters = [
                RegexFilter(
                    left_bound=f'/C_{class_num}_R_', 
                    right_bound='_emg.csv', 
                    values=[str(r) for r in range(self.num_reps)],
                    description='all reps'
                )
            ]
            num_reps = self.num_reps
        
        # Case 2: run_sgt OR manual step-through (i.e. auto_advance = False) → show single rep
        else:
            if rep_num is None:
                raise ValueError("No repetition number provided for recent data visualization.")
            regex_filters = [
                RegexFilter(
                    left_bound=f'/C_{class_num}_R_', 
                    right_bound='_emg.csv', 
                    values=[str(rep_num)],
                    description='single rep'
                )
            ]
            num_reps = 1

        self.offline_dh = OfflineDataHandler()
        self.offline_dh.get_data(folder_location=dataset_folder, regex_filters=regex_filters, delimiter=",")

        train_windows, train_metadata = self.offline_dh.parse_windows(
            self.WINDOW_SIZE,
            self.WINDOW_INCREMENT,
            metadata_operations=None
        )

        fe = FeatureExtractor()
        print("Extracting features")

        if chosen_features: 
            feature_list = chosen_features  
        else:
            print("No features selected, using default HTD features")
            feature_list = fe.get_feature_groups()["HTD"]
            
        feature_dict = fe.extract_features(feature_list=feature_list, windows=train_windows, array=False) 
        # feature_dict = {}
        # for feature in feature_list:
        #     train_features = fe.extract_features([feature], train_windows, array=True)
        #     feature_dict[feature] = train_features

        # Plot all features across selected repetitions
        self.feature_plot_thread = threading.Thread(
                                    target=self._plot_features_helper,
                                    args=(fe, feature_dict, sampling_time, class_name_to_num, num_reps, f"Features for class: {class_label}"),
                                    daemon=True
                                )
        self.feature_plot_thread.start()

    def _plot_features_helper(self, fe, fe_dict, sampling_time, class_name_dict, num_reps=1, title="Features"):
        fe.visualize_for_training(feature_dict=fe_dict, sampling_time=sampling_time, window_size=self.WINDOW_SIZE, window_increment=self.WINDOW_INCREMENT, num_reps=num_reps, class_names=class_name_dict, block=False, title=title)


    def play_collection_visual(self, media, active=True, next_motion=""):
        if active:
            timer_duration = media[-1]
            dpg.set_value("__dc_prompt", value="Current motion: "+media[1])
            dpg.set_item_width("__dc_prompt_spacer",width=self.video_player_width/2+30 - (7*len(media[1]))/2)
        else:
            timer_duration = self.rest_time
            dpg.set_value("__dc_prompt", value="Up next: "+next_motion)
            dpg.set_item_width("__dc_prompt_spacer",width=self.video_player_width/2+30 - (7*len("Up next: "+next_motion))/2)

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

