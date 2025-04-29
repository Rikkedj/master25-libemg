import serial
from libemg.environments.controllers import Controller
import time
import numpy as np
from pathlib import Path
import json

class Prosthesis:
    """
    Class for communicating with prosthetic device.
    """

    def __init__(self):
        """
        Initialize the Prosthesis object.
        """
        self.serial_port = "COM3"  # Default serial port (change as needed)
        self.baudrate = 9600
        self.timeout = 1
        self.connection = None

    def connect(self, serial_port, baudrate=9600, timeout=1):
        """
        Establish a connection to the prosthetic device.
        Parameters:
        ----------
            serial_port: The serial port to connect to (string).
            baudrate: The baud rate for the connection (int).
            timeout: Timeout for the connection (int).
        """
        try:
            self.connection = serial.Serial(
                port=serial_port,
                baudrate=baudrate,
                timeout=timeout
            )
            self.connection.flush()  # Clear input/output buffers
            print(f"Connected to prosthetic device on {self.serial_port}")
        except serial.SerialException as e:
            print(f"Error: Could not connect to prosthetic device: {e}")
            self.connection = None
    
    def disconnect(self):
        """
        Close the connection to the prosthetic device.
        """
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("Disconnected from prosthetic device.")
    
    def send_command(self, command):
        """
        Send a command to the prosthetic device.
        Parameters:
        ----------
            command: The command to send (string).
        """
        if self.connection and self.connection.is_open:
            try:
                packet = bytearray(command)
                #self.connection.write(command.encode('utf-8'))
                self.connection.write(packet)
                print(f"Sent command: {command}")
            except serial.SerialException as e:
                print(f"Error: Failed to send command: {e}")
        else:
            print("Error: No active connection to the prosthetic device.")


    def is_connected(self):
        """
        Check if the device is connected.
        ----------
        Return: True if connected, False otherwise.
        """
        return self.connection is not None and self.connection.is_open


class ActuatorFunctionSelection:
    def __init__(self, prosthesis: Prosthesis = None, controller: Controller = None):
        """
        Initialize the ActuatorFunctionSelection object.

        :param function_name: Name of the actuator function.
        :param function_code: Code representing the actuator function.
        """
        self.prosthesis = prosthesis
        self.controller = controller
        self.configurations = self.read_configurations()

    def get_prediction(self):
        pred = self.controller.get_data(['predictions'])
        if pred:
            return pred
        else:
            print("No prediction data available.")
            return None
    
    def read_configurations(self):
        config_path = Path('model/config.json').absolute()

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        try:
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
                print(f"Configuration loaded successfully from {config_path}")

            # Validate required keys
            required_keys = [
                'thr_angle_mf1', 'thr_angle_mf2', 'gain_mf1', 'gain_mf2',
                'window_size', 'window_increment', 'deadband'
            ]
            missing_keys = [key for key in required_keys if key not in config_data]
            if missing_keys:
                raise ValueError(f"Configuration file is missing required keys: {missing_keys}")

            return config_data

        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
        
    def get_motor_setpoints(self): 
        """
        Get the motor setpoints for the actuator function.
        """
        if self.prosthesis and self.prosthesis.is_connected() and self.controller:
            pred = self.get_prediction()
            if pred:
                mf1, mf2 = pred[0], pred[1]
                theta = np.arctan2(mf2, mf1)
                thr_mf1 = self.configurations['thr_angle_mf1']
                thr_mf2 = self.configurations['thr_angle_mf2']
                active_mf1, active_mf2 = self._compute_active_mf(theta, thr_mf1, thr_mf2)
                motor_setpoint = self.pred_to_motor_setpoints(active_mf1, active_mf2, mf1, mf2)
                return motor_setpoint
        else:
            print("No prosthesis connected.")
            return None
        
    def _compute_active_mf(self, theta, thresh_mf1, thresh_mf2):
        """Determine which degrees of freedom (DOFs) are active based on thresholds."""
        mf1_active, mf2_active = 0, 0
        rad2deg = np.rad2deg(theta)

        #if theta > 0:  # Upper half
        if thresh_mf1 < abs(rad2deg) < (180 - thresh_mf1):
            dof2_active = 1
        if abs(rad2deg) < (90 - thresh_mf2) or abs(rad2deg) > (90 + thresh_mf2):
            dof1_active = 1
        # else:  # Lower half
        #     if (-180 + thresh_dof1) < rad2deg < -thresh_dof1:
        #         dof2_active = 1
        #     if rad2deg < (-90 - thresh_dof2) or rad2deg > (-90 + thresh_dof2):
        #         dof1_active = 1

        return dof1_active, dof2_active
    
    def determine_actuator_function(self):
        """
        Determine the actuator function based on the prediction data.
        """
        prediction = self.get_prediction()  
    
    def pred_to_motor_setpoints(mf1_active, mf2_active, mf1, mf2):
        """
        Convert predicted class to motor setpoints.
        ------------------------------------------
        Parameters:
            mf1_active (bool): Whether motor function 1 is active.
            mf2_active (bool): Whether motor function 2 is active.
            
        Returns:
            motor_setpoint (list): Motor setpoint values.
        """   
        motor_setpoint = [0, 0, 0, 0] # rest
        max_value = 5

        if mf1_active:
            motor_setpoint[0 if mf1 > 0 else 1] = int(min(abs(mf1 / max_value * 255), 255))
        if mf2_active:
            motor_setpoint[2 if mf2 > 0 else 3] = int(min(abs(mf2 / max_value * 255), 255))

        return motor_setpoint

    def send_setpoint(self):
        """Send the motor setpoint to the prosthesis."""
        last_prediction = None  # Store the last prediction to avoid redundant transmissions
        if self.prosthesis and self.prosthesis.is_connected():
            prediction = self.get_prediction()
            if prediction != last_prediction:
                motor_setpoint = self.pred_to_motor_setpoints()
                self.prosthesis.send_command(motor_setpoint)
                last_prediction = prediction