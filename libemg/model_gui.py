import dearpygui.dearpygui as dpg
import time
import inspect
from libemg._gui._model_config_panel import ModelConfigPanel # Could remove this to my own folder or something to make it clear its made by me


class ML_GUI:
    '''
    The GUI for configuring the prediction model, either the classifier or the regressor. Mostly like the GUI for configuring the training model, but with some differences.
    '''
    def __init__(self, 
                 online_data_handler, 
                 args,
                 axis_images=None,
                 model_str=None,
                 width=1700,
                 height=1080,
                 debug=False,
                 plot_width = 500,
                 plot_height = 500,
                 regression_selected=False,
                 clean_up_on_kill=False):
        
        self.width = width
        self.height = height
        self.debug = debug # Usikker på hva denne er til, kan være den ikke trengs
        self.online_data_handler = online_data_handler
        self.axis_images = axis_images
        self.model_str = model_str
        self.args = args
        self.window = None
        self.plot_width = plot_width
        self.plot_height = plot_height
        self.regression_selected = regression_selected # Bool that tells if the regression model is selected, gotten from function in TrainingProtocol
        self.clean_up_on_kill = clean_up_on_kill
        #self.initialize_ui()
        #self.window.mainloop()

    def start_gui(self):
        """
        Opens the Model Configuration GUI
        """
        self._window_init(self.width, self.height, self.debug)

    def _window_init(self, width, height, debug=False):
        dpg.create_context()
        dpg.create_viewport(title="Configure Machine Learning Model",
                            width=width,
                            height=height)
        dpg.setup_dearpygui()
        

        self._file_menu_init()

        dpg.show_viewport()
        dpg.set_exit_callback(self._on_window_close)
        
        if debug:
            dpg.configure_app(manual_callback_management=True)
            while dpg.is_dearpygui_running():
                jobs = dpg.get_callback_queue()
                dpg.run_callbacks(jobs)
                dpg.render_dearpygui_frame()
        else:
            dpg.start_dearpygui()
        
        dpg.start_dearpygui()
        dpg.destroy_context()

    def _file_menu_init(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="Exit"):
                dpg.add_menu_item(label="Exit", callback=self._exit_window_callback) # Need to add a callback to close the window
                
            with dpg.menu(label="Model"):
                dpg.add_menu_item(label="Configure Model", callback=self._model_config_callback, show=True)
                
    def _model_config_callback(self):
        panel_arguments = list(inspect.signature(ModelConfigPanel.__init__).parameters)
        passed_arguments = {i: self.args[i] for i in self.args.keys() if i in panel_arguments}
        self.mcp = ModelConfigPanel(**passed_arguments, gui=self)
        self.mcp.spawn_configuration_window()

    def _exit_window_callback(self):
        #self.clean_up_on_kill = True
        dpg.stop_dearpygui()

    def _on_window_close(self):
        #if self.clean_up_on_kill:
        print("Window is closing. Performing clean-up...")
        if 'streamer' in self.args.keys():
            self.args['streamer'].signal.set()
        time.sleep(3)
    