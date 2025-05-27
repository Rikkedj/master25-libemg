import numpy as np
import time
import csv

class TrainingPrompt:
    def __init__(self, motor_functions=None, axis_media=None, rise_duration=1, steady_state_duration=1, amplitude=1, rest_between_reps=1, num_reps=1, sampling_rate=24):
        """
        Initializes the training protocol with default parameters.
        Parameters:
        -------------
        motor_functions (dict): Dictionary of motor functions to be trained as keys, and corresponding axis as value. Default is None.
        axis_media (dict): Dictionary mapping compass directions to media (image or video). Default is None.
        ramp_duration (float): Duration of the ramp-up phase in seconds.
        steady_state_duration (float): Duration of the steady state hold at endpoint in seconds.
        amplitude (float): Amplitude of the steady state.
        rest_between_reps (float): Duration of the rest period between repetitions in seconds.
        num_reps (int): Number of repetitions.
        sampling_rate (int): Sampling rate in Hz. Should be same as the FPS when making animation. 

        """
        self.rise_duration = rise_duration
        self.steady_state_duration = steady_state_duration
        self.amplitude = amplitude
        self.rest_between_reps = rest_between_reps
        self.num_reps = num_reps
        self.sampling_rate = sampling_rate
        
        self.motor_functions = motor_functions if motor_functions is not None else {'MF1': (1,0), 'MF2': (0,1)}
        self.axis_media = axis_media

    
    def generate_training_signal(self):
        """
        Generates a sawtooth-like training signal with a ramp-up, steady state, and rest time between reps.

        Returns:
        numpy.ndarray: The generated training signal.
        """
       
        # Number of samples for rise and rest
        rise_samples = int(self.rise_duration * self.sampling_rate)
        hold_samples = int(self.steady_state_duration * self.sampling_rate)
        rest_samples = int(self.rest_between_reps * self.sampling_rate)

        # Create one cycle: rise + rest
        rise_part = np.linspace(0, self.amplitude, rise_samples, endpoint=False)
        hold_part = np.full(hold_samples, self.amplitude)
        rest_part = np.zeros(rest_samples)

        # One full cycle
        cycle = np.concatenate([rise_part, hold_part, rest_part])

        # Repeat the cycle
        signal = np.tile(cycle, self.num_reps)

        # Time vector
        total_duration = self.num_reps * (self.rise_duration + self.steady_state_duration + self.rest_between_reps)
        time_vec = np.linspace(0, total_duration, len(signal), endpoint=False)

        return signal, time_vec
    
    def show_training_plot(self, motor_function):
        # Initialize plot with center plot showing the training signal and the other plots showing the motor function medias
        pass  # Placeholder for the function to show the training plot. Could use the PredictionPLotter class
    
    def _plot_helper(self, motor_function, signal, time_vec):
        # Use this so if block is True, the plotting happens in new process
        pass

    def collect_training_data(self, output_folder, online_data_handler):
        """
        Collects training data from the online data handler and saves it to the specified folder.
        """
        self.odh = online_data_handler
        self.output_folder = output_folder
        data_buf = []
        self.current_rep = 0
        start_time = time.perf_counter_ns()
        if self.current_rep < self.num_reps:
            while (time.perf_counter_ns() - start_time)/1e9 < (self.rise_duration+self.steady_state_duration):
                # Collect data from the online data handler
                vals, count = self.odh.get_data()
                data_buf.append(vals)
                time.sleep(1/self.sampling_rate)
            
            self.current_rep += 1
            # Wait for the rest time before the next repetition
            time.sleep(self.rest_between_reps)

        self.save_data(data_buf, self.output_folder)

    def save_data(self, output_folder):
        file_parts = output_folder.split('.')
        
        for mod in self.rep_buffer:
            filename = file_parts[0] + "_" + mod + "." + file_parts[1]
            data = np.vstack(self.rep_buffer[mod])[::-1,:]
            if data.size == 0:
                raise ConnectionError('Attempting to store data, but received 0 samples during repetition, suggesting that the data stream from the device has been interrupted. Please check the device connection and verify that previous files are not missing samples.')
            with open(filename, "w", newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for row in data:
                    writer.writerow(row)


class TrainingProtocol:
    def __init__(self, motor_functions=None, axis_media=None, rise_duration=1, steady_state_duration=1, amplitude=1, rest_between_reps=1, num_reps=1, sampling_rate=24, time_between_sets=1):
        """
        Initializes the training protocol with default parameters.
        Parameters:
        -------------
        motor_functions (dict): Dictionary of motor functions to be trained as keys, and corresponding axis as value. Default is None.
        axis_media (dict): Dictionary mapping compass directions to media (image or video). Default is None.
        ramp_duration (float): Duration of the ramp-up phase in seconds.
        steady_state_duration (float): Duration of the steady state hold at endpoint in seconds.
        amplitude (float): Amplitude of the steady state.
        rest_between_reps (float): Duration of the rest period between repetitions in seconds.
        num_reps (int): Number of repetitions.
        sampling_rate (int): Sampling rate in Hz. Should be same as the FPS when making animation. 

        """
        self.rise_duration = rise_duration
        self.steady_state_duration = steady_state_duration
        self.amplitude = amplitude
        self.rest_between_reps = rest_between_reps
        self.num_reps = num_reps
        self.sampling_rate = sampling_rate
        
        self.motor_functions = motor_functions if motor_functions is not None else {'MF1': (1,0), 'MF2': (0,1)}
        self.axis_media = axis_media

        self.training_data_folder = './data/'
        self.feature_list = None
        self.regression_selected = False
    
    def generate_training_signal(self):
        """
        Generates a sawtooth-like training signal with a ramp-up, steady state, and rest time between reps.

        Returns:
        numpy.ndarray: The generated training signal.
        """
       
        # Number of samples for rise and rest
        rise_samples = int(self.rise_duration * self.sampling_rate)
        hold_samples = int(self.steady_state_duration * self.sampling_rate)
        rest_samples = int(self.rest_between_reps * self.sampling_rate)

        # Create one cycle: rise + rest
        rise_part = np.linspace(0, self.amplitude, rise_samples, endpoint=False)
        hold_part = np.full(hold_samples, self.amplitude)
        rest_part = np.zeros(rest_samples)

        # One full cycle
        cycle = np.concatenate([rise_part, hold_part, rest_part])

        # Repeat the cycle
        signal = np.tile(cycle, self.num_reps)

        # Time vector
        total_duration = self.num_reps * (self.rise_duration + self.steady_state_duration + self.rest_between_reps)
        time_vec = np.linspace(0, total_duration, len(signal), endpoint=False)

        return signal, time_vec
    